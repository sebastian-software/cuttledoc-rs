import AVFoundation
import CoreMedia
import Darwin
import Foundation
import Speech

public typealias CuttledocSpeechUpdateCallback = @convention(c) (
    UnsafeMutableRawPointer?,
    UnsafePointer<CChar>?
) -> Void

private let speechErrorDomain = "CuttledocSpeechShim"
private func makeJSONEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return encoder
}

private final class BlockingBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value?

    func store(_ value: Value) {
        lock.lock()
        self.value = value
        lock.unlock()
    }

    func take() -> Value? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private final class CallbackContext: @unchecked Sendable {
    let raw: UnsafeMutableRawPointer?

    init(_ raw: UnsafeMutableRawPointer?) {
        self.raw = raw
    }
}

private struct MillisecondRange: Encodable {
    let startMs: Int64
    let endMs: Int64
}

private struct Segment: Encodable {
    let text: String
    let startMs: Int64
    let endMs: Int64
    let confidence: Double?
}

private struct Update: Encodable {
    let sequence: UInt64
    let kind: String
    let stability: String?
    let replaceRange: MillisecondRange
    let text: String
    let segments: [Segment]
}

private struct ResultStats: Encodable {
    let firstResultMs: Double?
    let updateCount: Int
    let volatileUpdateCount: Int
    let finalUpdateCount: Int
    let revokeCount: Int
    let elapsedMs: Double
}

private struct SessionMetadata: Encodable {
    let locale: String
    let sampleRateHz: UInt32
    let assetStatusBefore: String
    let assetStatusAfter: String
    let installationRequestReturned: Bool
    let reservationOwnedBySession: Bool
    let bundleIdentifier: String
    let executablePath: String
}

private struct LocaleInventory: Encodable {
    let supported: [String]
    let installed: [String]
    let reserved: [String]
    let maximumReservedLocales: Int
}

private final class SpeechSession: @unchecked Sendable {
    private enum State {
        case active
        case finishing
        case finished
        case cancelled
        case closed
    }

    private let analyzer: SpeechAnalyzer
    private let transcriber: SpeechTranscriber
    private let format: AVAudioFormat
    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private let stream: AsyncStream<AnalyzerInput>
    private let callback: CuttledocSpeechUpdateCallback
    private let callbackContext: UnsafeMutableRawPointer?
    private let locale: Locale
    private let ownsReservation: Bool
    private let startedAt = DispatchTime.now().uptimeNanoseconds
    private let lock = NSLock()
    private var frameCursor: Int64 = 0
    private var state: State = .active
    private var analysisTask: Task<Void, Error>?
    private var resultTask: Task<ResultStats, Error>?

    private init(
        analyzer: SpeechAnalyzer,
        transcriber: SpeechTranscriber,
        format: AVAudioFormat,
        continuation: AsyncStream<AnalyzerInput>.Continuation,
        stream: AsyncStream<AnalyzerInput>,
        callback: @escaping CuttledocSpeechUpdateCallback,
        callbackContext: UnsafeMutableRawPointer?,
        locale: Locale,
        ownsReservation: Bool
    ) {
        self.analyzer = analyzer
        self.transcriber = transcriber
        self.format = format
        self.continuation = continuation
        self.stream = stream
        self.callback = callback
        self.callbackContext = callbackContext
        self.locale = locale
        self.ownsReservation = ownsReservation
    }

    static func make(
        localeIdentifier: String,
        sampleRate: UInt32,
        callback: @escaping CuttledocSpeechUpdateCallback,
        callbackContext: UnsafeMutableRawPointer?
    ) async throws -> (SpeechSession, SessionMetadata) {
        guard sampleRate > 0 else {
            throw speechError(10, "sample rate must be positive")
        }
        guard let locale = await SpeechTranscriber.supportedLocale(
            equivalentTo: Locale(identifier: localeIdentifier)
        ) else {
            throw speechError(11, "locale \(localeIdentifier) is not supported")
        }

        let previouslyReserved = await AssetInventory.reservedLocales.contains {
            $0.identifier == locale.identifier
        }
        let reservationSucceeded = try await AssetInventory.reserve(locale: locale)
        let ownsReservation = reservationSucceeded && !previouslyReserved

        do {
            let transcriber = SpeechTranscriber(
                locale: locale,
                transcriptionOptions: [],
                reportingOptions: [.volatileResults],
                attributeOptions: [.audioTimeRange, .transcriptionConfidence]
            )
            let statusBefore = await AssetInventory.status(forModules: [transcriber])
            let request = try await AssetInventory.assetInstallationRequest(
                supporting: [transcriber]
            )
            if let request {
                try await request.downloadAndInstall()
            }
            let statusAfter = await AssetInventory.status(forModules: [transcriber])

            guard let format = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false
            ) else {
                throw speechError(12, "could not create mono Int16 PCM format")
            }

            var capturedContinuation: AsyncStream<AnalyzerInput>.Continuation?
            let stream = AsyncStream<AnalyzerInput>(
                bufferingPolicy: .bufferingOldest(64)
            ) { continuation in
                capturedContinuation = continuation
            }
            guard let continuation = capturedContinuation else {
                throw speechError(13, "could not create PCM input stream")
            }

            let analyzer = SpeechAnalyzer(
                modules: [transcriber],
                options: .init(
                    priority: .userInitiated,
                    modelRetention: .whileInUse
                )
            )
            try await analyzer.prepareToAnalyze(in: format)

            let session = SpeechSession(
                analyzer: analyzer,
                transcriber: transcriber,
                format: format,
                continuation: continuation,
                stream: stream,
                callback: callback,
                callbackContext: callbackContext,
                locale: locale,
                ownsReservation: ownsReservation
            )
            session.startTasks()

            return (
                session,
                SessionMetadata(
                    locale: locale.identifier,
                    sampleRateHz: sampleRate,
                    assetStatusBefore: assetStatus(statusBefore),
                    assetStatusAfter: assetStatus(statusAfter),
                    installationRequestReturned: request != nil,
                    reservationOwnedBySession: ownsReservation,
                    bundleIdentifier: Bundle.main.bundleIdentifier ?? "none",
                    executablePath: ProcessInfo.processInfo.arguments.first ?? ""
                )
            )
        } catch {
            if ownsReservation {
                _ = await AssetInventory.release(reservedLocale: locale)
            }
            throw error
        }
    }

    func push(_ samples: UnsafePointer<Float>, count: Int) -> Int32 {
        guard count > 0, count <= Int(UInt32.max) else {
            return 2
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(count)
        ), let destination = buffer.int16ChannelData?[0] else {
            return 1
        }
        buffer.frameLength = AVAudioFrameCount(count)
        for index in 0..<count {
            let sample = max(-1, min(1, samples[index]))
            destination[index] = Int16(
                (sample * Float(Int16.max)).rounded()
            )
        }

        lock.lock()
        defer { lock.unlock() }
        guard state == .active else {
            return 3
        }

        let start = frameCursor
        let input = AnalyzerInput(
            buffer: buffer,
            bufferStartTime: CMTime(
                value: start,
                timescale: CMTimeScale(format.sampleRate)
            )
        )
        switch continuation.yield(input) {
        case .enqueued:
            frameCursor += Int64(count)
            return 0
        case .dropped:
            return 4
        case .terminated:
            return 3
        @unknown default:
            return 1
        }
    }

    func finish() async throws -> ResultStats {
        let wasActive = lock.withLock {
            guard state == .active else {
                return false
            }
            state = .finishing
            return true
        }
        guard wasActive else {
            throw speechError(14, "session is not active")
        }
        continuation.finish()

        guard let analysisTask, let resultTask else {
            throw speechError(15, "session tasks were not started")
        }
        try await analysisTask.value
        let stats = try await resultTask.value

        lock.withLock {
            state = .finished
        }
        return stats
    }

    func cancel() async {
        let shouldCancel = lock.withLock {
            guard state != .closed && state != .cancelled else {
                return false
            }
            state = .cancelled
            return true
        }
        guard shouldCancel else {
            return
        }
        continuation.finish()

        await analyzer.cancelAndFinishNow()
        analysisTask?.cancel()
        resultTask?.cancel()
        _ = try? await analysisTask?.value
        _ = try? await resultTask?.value
    }

    func close() async {
        let (needsCancellation, alreadyClosed) = lock.withLock {
            (state == .active || state == .finishing, state == .closed)
        }
        if alreadyClosed {
            return
        }
        if needsCancellation {
            await cancel()
        }
        if ownsReservation {
            _ = await AssetInventory.release(reservedLocale: locale)
        }
        lock.withLock {
            state = .closed
        }
    }

    private func startTasks() {
        let analyzer = self.analyzer
        let stream = self.stream
        analysisTask = Task {
            if let lastSample = try await analyzer.analyzeSequence(stream) {
                try await analyzer.finalizeAndFinish(through: lastSample)
            } else {
                await analyzer.cancelAndFinishNow()
            }
        }

        let transcriber = self.transcriber
        let callback = self.callback
        let context = self.callbackContext
        let startedAt = self.startedAt
        resultTask = Task {
            var sequence: UInt64 = 0
            var firstResultMs: Double?
            var volatileCount = 0
            var finalCount = 0
            var revokeCount = 0

            for try await result in transcriber.results {
                sequence += 1
                let now = DispatchTime.now().uptimeNanoseconds
                if firstResultMs == nil {
                    firstResultMs = milliseconds(from: startedAt, to: now)
                }
                let text = String(result.text.characters)
                let isRevoke = text.isEmpty
                if isRevoke {
                    revokeCount += 1
                } else if result.isFinal {
                    finalCount += 1
                } else {
                    volatileCount += 1
                }

                let update = Update(
                    sequence: sequence,
                    kind: isRevoke ? "revoke" : "replace",
                    stability: isRevoke ? nil : (result.isFinal ? "final" : "volatile"),
                    replaceRange: milliseconds(result.range),
                    text: text,
                    segments: isRevoke ? [] : segments(from: result)
                )
                let data = try makeJSONEncoder().encode(update)
                guard let json = String(data: data, encoding: .utf8) else {
                    throw speechError(16, "could not encode update as UTF-8")
                }
                json.withCString { pointer in
                    callback(context, pointer)
                }
            }

            return ResultStats(
                firstResultMs: firstResultMs,
                updateCount: Int(sequence),
                volatileUpdateCount: volatileCount,
                finalUpdateCount: finalCount,
                revokeCount: revokeCount,
                elapsedMs: milliseconds(
                    from: startedAt,
                    to: DispatchTime.now().uptimeNanoseconds
                )
            )
        }
    }
}

@_cdecl("cuttledoc_speech_locale_inventory")
public func cuttledoc_speech_locale_inventory(
    _ output: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    output.pointee = nil
    let result: Result<LocaleInventory, Error> = blocking {
        async let supported = SpeechTranscriber.supportedLocales
        async let installed = SpeechTranscriber.installedLocales
        async let reserved = AssetInventory.reservedLocales
        return await LocaleInventory(
            supported: supported.map(\.identifier).sorted(),
            installed: installed.map(\.identifier).sorted(),
            reserved: reserved.map(\.identifier).sorted(),
            maximumReservedLocales: AssetInventory.maximumReservedLocales
        )
    }
    switch result {
    case let .success(inventory):
        return writeJSON(inventory, to: output)
    case let .failure(error):
        output.pointee = strdup(String(describing: error))
        return 1
    }
}

@_cdecl("cuttledoc_speech_session_create")
public func cuttledoc_speech_session_create(
    _ locale: UnsafePointer<CChar>,
    _ sampleRate: UInt32,
    _ callback: @escaping CuttledocSpeechUpdateCallback,
    _ callbackContext: UnsafeMutableRawPointer?,
    _ metadata: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOutput: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> UnsafeMutableRawPointer? {
    metadata.pointee = nil
    errorOutput.pointee = nil
    guard let locale = String(validatingUTF8: locale) else {
        errorOutput.pointee = strdup("locale is not valid UTF-8")
        return nil
    }

    let sendableContext = CallbackContext(callbackContext)
    let result: Result<(SpeechSession, SessionMetadata), Error> = blocking {
        try await SpeechSession.make(
            localeIdentifier: locale,
            sampleRate: sampleRate,
            callback: callback,
            callbackContext: sendableContext.raw
        )
    }
    switch result {
    case let .success((session, sessionMetadata)):
        guard writeJSON(sessionMetadata, to: metadata) == 0 else {
            errorOutput.pointee = strdup("could not encode session metadata")
            return nil
        }
        return Unmanaged.passRetained(session).toOpaque()
    case let .failure(error):
        errorOutput.pointee = strdup(String(describing: error))
        return nil
    }
}

@_cdecl("cuttledoc_speech_session_push_pcm_f32")
public func cuttledoc_speech_session_push_pcm_f32(
    _ handle: UnsafeMutableRawPointer,
    _ samples: UnsafePointer<Float>,
    _ sampleCount: UInt32
) -> Int32 {
    speechSession(handle).push(samples, count: Int(sampleCount))
}

@_cdecl("cuttledoc_speech_session_finish")
public func cuttledoc_speech_session_finish(
    _ handle: UnsafeMutableRawPointer,
    _ summary: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOutput: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    summary.pointee = nil
    errorOutput.pointee = nil
    let session = speechSession(handle)
    let result: Result<ResultStats, Error> = blocking {
        try await session.finish()
    }
    switch result {
    case let .success(stats):
        return writeJSON(stats, to: summary)
    case let .failure(error):
        errorOutput.pointee = strdup(String(describing: error))
        return 1
    }
}

@_cdecl("cuttledoc_speech_session_cancel")
public func cuttledoc_speech_session_cancel(
    _ handle: UnsafeMutableRawPointer
) {
    let session = speechSession(handle)
    let _: Result<Void, Error> = blocking {
        await session.cancel()
    }
}

@_cdecl("cuttledoc_speech_session_destroy")
public func cuttledoc_speech_session_destroy(
    _ handle: UnsafeMutableRawPointer
) {
    let session = speechSession(handle)
    let _: Result<Void, Error> = blocking {
        await session.close()
    }
    Unmanaged<SpeechSession>.fromOpaque(handle).release()
}

@_cdecl("cuttledoc_speech_transcribe_file")
public func cuttledoc_speech_transcribe_file(
    _ path: UnsafePointer<CChar>,
    _ output: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    output.pointee = nil
    guard let path = String(validatingUTF8: path) else {
        output.pointee = strdup("input path is not valid UTF-8")
        return 2
    }

    let result: Result<String, Error> = blocking {
        try await transcribe(path: path)
    }
    switch result {
    case let .success(text):
        output.pointee = strdup(text)
        return 0
    case let .failure(error):
        output.pointee = strdup(String(describing: error))
        return 1
    }
}

@_cdecl("cuttledoc_speech_free_string")
public func cuttledoc_speech_free_string(_ value: UnsafeMutablePointer<CChar>?) {
    free(value)
}

private func blocking<Value: Sendable>(
    _ operation: @escaping @Sendable () async throws -> Value
) -> Result<Value, Error> {
    let completion = DispatchSemaphore(value: 0)
    let box = BlockingBox<Result<Value, Error>>()
    Task {
        do {
            box.store(.success(try await operation()))
        } catch {
            box.store(.failure(error))
        }
        completion.signal()
    }
    completion.wait()
    return box.take() ?? .failure(
        speechError(17, "Swift task completed without a result")
    )
}

private func speechSession(_ handle: UnsafeMutableRawPointer) -> SpeechSession {
    Unmanaged<SpeechSession>.fromOpaque(handle).takeUnretainedValue()
}

private func writeJSON<Value: Encodable>(
    _ value: Value,
    to output: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    do {
        let data = try makeJSONEncoder().encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            output.pointee = strdup("JSON is not valid UTF-8")
            return 1
        }
        output.pointee = strdup(json)
        return 0
    } catch {
        output.pointee = strdup(String(describing: error))
        return 1
    }
}

private func segments(from result: SpeechTranscriber.Result) -> [Segment] {
    result.text.runs.compactMap { run in
        let text = String(result.text[run.range].characters)
        guard !text.isEmpty else {
            return nil
        }
        let timeRange =
            run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self]
            ?? result.range
        return Segment(
            text: text,
            startMs: milliseconds(timeRange).startMs,
            endMs: milliseconds(timeRange).endMs,
            confidence:
                run[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self]
        )
    }
}

private func milliseconds(_ range: CMTimeRange) -> MillisecondRange {
    let start = CMTimeGetSeconds(range.start)
    let end = CMTimeGetSeconds(CMTimeRangeGetEnd(range))
    return MillisecondRange(
        startMs: finiteMilliseconds(start),
        endMs: finiteMilliseconds(end)
    )
}

private func finiteMilliseconds(_ seconds: Double) -> Int64 {
    guard seconds.isFinite else {
        return 0
    }
    return Int64((seconds * 1_000).rounded())
}

private func milliseconds(from start: UInt64, to end: UInt64) -> Double {
    Double(end - start) / 1_000_000
}

private func assetStatus(_ status: AssetInventory.Status) -> String {
    switch status {
    case .unsupported: "unsupported"
    case .supported: "supported"
    case .downloading: "downloading"
    case .installed: "installed"
    @unknown default: "unknown"
    }
}

private func speechError(_ code: Int, _ message: String) -> NSError {
    NSError(
        domain: speechErrorDomain,
        code: code,
        userInfo: [NSLocalizedDescriptionKey: message]
    )
}

private func transcribe(path: String) async throws -> String {
    guard let locale = await SpeechTranscriber.supportedLocale(
        equivalentTo: Locale(identifier: "en-US")
    ) else {
        throw speechError(1, "en-US is not supported")
    }

    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    if let request = try await AssetInventory.assetInstallationRequest(
        supporting: [transcriber]
    ) {
        try await request.downloadAndInstall()
    }

    let audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    async let transcript: String = try transcriber.results.reduce("") {
        text,
        result in
        text + String(result.text.characters)
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])
    if let lastSample = try await analyzer.analyzeSequence(from: audioFile) {
        try await analyzer.finalizeAndFinish(through: lastSample)
    } else {
        await analyzer.cancelAndFinishNow()
    }

    return try await transcript
}
