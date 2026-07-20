import AVFoundation
import Darwin
import Foundation

private let ttsErrorDomain = "CuttledocTtsShim"

private func ttsError(_ code: Int, _ message: String) -> NSError {
    NSError(
        domain: ttsErrorDomain,
        code: code,
        userInfo: [NSLocalizedDescriptionKey: message]
    )
}

private func makeJSONEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return encoder
}

private struct VoiceRecord: Encodable {
    let identifier: String
    let name: String
    let language: String
    let qualityRawValue: Int
    let genderRawValue: Int
}

private struct VoiceInventory: Encodable {
    let requestedLocale: String
    let voices: [VoiceRecord]
}

private struct SessionMetadata: Encodable {
    let requestedLocale: String
    let voice: VoiceRecord
    let rate: Float
    let pitchMultiplier: Float
    let volume: Float
    let outputOwnership: String
}

private struct SynthesisStats: Encodable {
    let status: String
    let voiceIdentifier: String
    let voiceName: String
    let voiceLanguage: String
    let sampleRateHz: UInt32?
    let channelCount: UInt32
    let sampleFormat: String
    let sampleCount: UInt64
    let audioDurationMs: Double?
    let chunkCount: Int
    let firstAudioMs: Double?
    let elapsedMs: Double
    let cancellationLatencyMs: Double?
}

private struct CapturedAudio {
    let samples: [Float]
    let sampleRateHz: UInt32
    let chunkCount: Int
    let firstAudioMs: Double?
}

private final class TtsSession: NSObject, AVSpeechSynthesizerDelegate,
    @unchecked Sendable
{
    private let synthesizer = AVSpeechSynthesizer()
    private let voice: AVSpeechSynthesisVoice
    private let requestedLocale: String
    private let lock = NSLock()
    private var busy = false
    private var closed = false
    private var cancelRequested = false
    private var cancelRequestedAt: UInt64?
    private var cancellationObservedAt: UInt64?
    private var completionSignaled = false
    private var completion = DispatchSemaphore(value: 0)
    private var synthesisStartedAt: UInt64 = 0
    private var firstAudioAt: UInt64?
    private var outputSampleRate: Double?
    private var outputSamples: [Float] = []
    private var outputChunkCount = 0
    private var outputError: Error?
    private var diagnostics: [String] = []

    init(requestedLocale: String, voice: AVSpeechSynthesisVoice) {
        self.requestedLocale = requestedLocale
        self.voice = voice
        super.init()
        synthesizer.delegate = self
    }

    var metadata: SessionMetadata {
        SessionMetadata(
            requestedLocale: requestedLocale,
            voice: voiceRecord(voice),
            rate: AVSpeechUtteranceDefaultSpeechRate,
            pitchMultiplier: 1,
            volume: 1,
            outputOwnership: "caller-owned mono f32 PCM copied at the C ABI"
        )
    }

    func synthesize(_ text: String) -> Result<(CapturedAudio, SynthesisStats), Error> {
        lock.lock()
        if closed {
            lock.unlock()
            return .failure(ttsError(3, "session is closed"))
        }
        if busy {
            lock.unlock()
            return .failure(ttsError(4, "session is busy"))
        }
        busy = true
        cancelRequested = false
        cancelRequestedAt = nil
        cancellationObservedAt = nil
        completionSignaled = false
        completion = DispatchSemaphore(value: 0)
        synthesisStartedAt = DispatchTime.now().uptimeNanoseconds
        firstAudioAt = nil
        outputSampleRate = nil
        outputSamples = []
        outputChunkCount = 0
        outputError = nil
        diagnostics = []
        lock.unlock()

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1
        utterance.volume = 1
        utterance.preUtteranceDelay = 0
        utterance.postUtteranceDelay = 0
        synthesizer.write(utterance) { [weak self] buffer in
            self?.receive(buffer)
        }
        recordDiagnostic("write-returned")
        while completion.wait(timeout: .now()) != .success {
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.01))
        }

        let endedAt = DispatchTime.now().uptimeNanoseconds
        lock.lock()
        let error = outputError
        let callerCancelled = cancelRequested
        let unexpectedlyCancelled =
            cancellationObservedAt != nil && !cancelRequested
        let samples = outputSamples
        let sampleRate = outputSampleRate
        let chunkCount = outputChunkCount
        let firstAudio = firstAudioAt
        let cancelAt = cancelRequestedAt
        let cancelObserved =
            cancellationObservedAt ?? (callerCancelled ? endedAt : nil)
        let capturedDiagnostics = diagnostics
        busy = false
        outputSamples = []
        lock.unlock()

        let elapsedMs = milliseconds(from: synthesisStartedAt, to: endedAt)
        let firstAudioMs = firstAudio.map {
            milliseconds(from: synthesisStartedAt, to: $0)
        }
        let cancellationLatencyMs: Double? =
            if let cancelAt, let cancelObserved {
                milliseconds(from: cancelAt, to: cancelObserved)
            } else {
                nil
            }

        if let error {
            return .failure(error)
        }
        if callerCancelled {
            return .failure(
                CancellationError(
                    stats: SynthesisStats(
                        status: "cancelled",
                        voiceIdentifier: voice.identifier,
                        voiceName: voice.name,
                        voiceLanguage: voice.language,
                        sampleRateHz: sampleRate.map { UInt32($0.rounded()) },
                        channelCount: 1,
                        sampleFormat: "f32le",
                        sampleCount: UInt64(samples.count),
                        audioDurationMs: durationMs(
                            sampleCount: samples.count,
                            sampleRate: sampleRate
                        ),
                        chunkCount: chunkCount,
                        firstAudioMs: firstAudioMs,
                        elapsedMs: elapsedMs,
                        cancellationLatencyMs: cancellationLatencyMs
                    )
                )
            )
        }
        if unexpectedlyCancelled {
            return .failure(
                ttsError(
                    12,
                    "synthesizer cancelled unexpectedly: " +
                    capturedDiagnostics.joined(separator: ", ")
                )
            )
        }
        guard let sampleRate, sampleRate > 0, !samples.isEmpty else {
            return .failure(ttsError(5, "synthesis produced no audio"))
        }
        let captured = CapturedAudio(
            samples: samples,
            sampleRateHz: UInt32(sampleRate.rounded()),
            chunkCount: chunkCount,
            firstAudioMs: firstAudioMs
        )
        return .success((
            captured,
            SynthesisStats(
                status: "ok",
                voiceIdentifier: voice.identifier,
                voiceName: voice.name,
                voiceLanguage: voice.language,
                sampleRateHz: captured.sampleRateHz,
                channelCount: 1,
                sampleFormat: "f32le",
                sampleCount: UInt64(samples.count),
                audioDurationMs: durationMs(
                    sampleCount: samples.count,
                    sampleRate: sampleRate
                ),
                chunkCount: chunkCount,
                firstAudioMs: firstAudioMs,
                elapsedMs: elapsedMs,
                cancellationLatencyMs: nil
            )
        ))
    }

    func cancel() {
        lock.lock()
        guard busy, !cancelRequested else {
            lock.unlock()
            return
        }
        cancelRequested = true
        cancelRequestedAt = DispatchTime.now().uptimeNanoseconds
        lock.unlock()
        _ = synthesizer.stopSpeaking(at: .immediate)
        signalCompletion(cancelled: true)
    }

    func close() {
        lock.lock()
        closed = true
        let shouldCancel = busy
        lock.unlock()
        if shouldCancel {
            cancel()
        }
    }

    func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        recordDiagnostic("delegate-finished")
        signalCompletion(cancelled: false)
    }

    func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        recordDiagnostic("delegate-cancelled")
        signalCompletion(cancelled: true)
    }

    private func receive(_ buffer: AVAudioBuffer) {
        guard let pcm = buffer as? AVAudioPCMBuffer else {
            fail(ttsError(6, "AVSpeechSynthesizer returned a non-PCM buffer"))
            return
        }
        recordDiagnostic(
            "buffer-\(pcm.frameLength)-\(pcm.format.commonFormat.rawValue)-" +
            "\(pcm.format.channelCount)-\(pcm.format.sampleRate)"
        )
        if pcm.frameLength == 0 {
            signalCompletion(cancelled: false)
            return
        }
        do {
            let chunk = try monoFloatSamples(pcm)
            lock.lock()
            if firstAudioAt == nil {
                firstAudioAt = DispatchTime.now().uptimeNanoseconds
            }
            if let currentRate = outputSampleRate {
                if abs(currentRate - pcm.format.sampleRate) > 0.5 {
                    outputError = ttsError(7, "voice output sample rate changed")
                }
            } else {
                outputSampleRate = pcm.format.sampleRate
            }
            outputSamples.append(contentsOf: chunk)
            outputChunkCount += 1
            let mustStop = outputError != nil
            lock.unlock()
            if mustStop {
                _ = synthesizer.stopSpeaking(at: .immediate)
                signalCompletion(cancelled: false)
            }
        } catch {
            fail(error)
        }
    }

    private func fail(_ error: Error) {
        lock.lock()
        outputError = error
        lock.unlock()
        _ = synthesizer.stopSpeaking(at: .immediate)
        signalCompletion(cancelled: false)
    }

    private func signalCompletion(cancelled: Bool) {
        lock.lock()
        guard busy, !completionSignaled else {
            lock.unlock()
            return
        }
        if cancelled {
            cancellationObservedAt = DispatchTime.now().uptimeNanoseconds
        }
        completionSignaled = true
        let semaphore = completion
        lock.unlock()
        semaphore.signal()
    }

    private func recordDiagnostic(_ value: String) {
        lock.lock()
        if diagnostics.count < 8 {
            diagnostics.append(value)
        }
        lock.unlock()
    }
}

private struct CancellationError: Error {
    let stats: SynthesisStats
}

private func voiceRecord(_ voice: AVSpeechSynthesisVoice) -> VoiceRecord {
    VoiceRecord(
        identifier: voice.identifier,
        name: voice.name,
        language: voice.language,
        qualityRawValue: voice.quality.rawValue,
        genderRawValue: voice.gender.rawValue
    )
}

private func monoFloatSamples(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
    let frames = Int(buffer.frameLength)
    let channels = Int(buffer.format.channelCount)
    guard frames > 0, channels > 0 else {
        return []
    }
    var output = [Float](repeating: 0, count: frames)
    switch buffer.format.commonFormat {
    case .pcmFormatFloat32:
        guard let data = buffer.floatChannelData else {
            throw ttsError(
                8,
                "float PCM has no channel data: \(buffer.format)"
            )
        }
        for channel in 0..<channels {
            for frame in 0..<frames {
                output[frame] += data[channel][frame] / Float(channels)
            }
        }
    case .pcmFormatInt16:
        guard let data = buffer.int16ChannelData else {
            throw ttsError(
                9,
                "int16 PCM has no channel data: \(buffer.format)"
            )
        }
        for channel in 0..<channels {
            for frame in 0..<frames {
                output[frame] +=
                    Float(data[channel][frame]) / 32768 / Float(channels)
            }
        }
    case .pcmFormatInt32:
        guard let data = buffer.int32ChannelData else {
            throw ttsError(
                10,
                "int32 PCM has no channel data: \(buffer.format)"
            )
        }
        for channel in 0..<channels {
            for frame in 0..<frames {
                output[frame] +=
                    Float(data[channel][frame]) / 2_147_483_648 /
                    Float(channels)
            }
        }
    default:
        throw ttsError(
            11,
            "unsupported PCM format \(buffer.format.commonFormat.rawValue)"
        )
    }
    return output
}

private func milliseconds(from start: UInt64, to end: UInt64) -> Double {
    Double(end - start) / 1_000_000
}

private func durationMs(sampleCount: Int, sampleRate: Double?) -> Double? {
    guard let sampleRate, sampleRate > 0 else {
        return nil
    }
    return Double(sampleCount) * 1_000 / sampleRate
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

private func ttsSession(_ handle: UnsafeMutableRawPointer) -> TtsSession {
    Unmanaged<TtsSession>.fromOpaque(handle).takeUnretainedValue()
}

@_cdecl("cuttledoc_tts_voice_inventory")
public func cuttledoc_tts_voice_inventory(
    _ locale: UnsafePointer<CChar>,
    _ output: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOutput: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    output.pointee = nil
    errorOutput.pointee = nil
    guard let locale = String(validatingUTF8: locale) else {
        errorOutput.pointee = strdup("locale is not valid UTF-8")
        return 2
    }
    let prefix = locale.split(separator: "-").first.map(String.init)
    let voices = AVSpeechSynthesisVoice.speechVoices()
        .filter {
            $0.language == locale ||
            $0.language.split(separator: "-").first.map(String.init) == prefix
        }
        .map(voiceRecord)
        .sorted {
            ($0.language, $0.name, $0.identifier) <
            ($1.language, $1.name, $1.identifier)
        }
    let status = writeJSON(
        VoiceInventory(requestedLocale: locale, voices: voices),
        to: output
    )
    if status != 0 {
        errorOutput.pointee = output.pointee
        output.pointee = nil
    }
    return status
}

@_cdecl("cuttledoc_tts_session_create")
public func cuttledoc_tts_session_create(
    _ locale: UnsafePointer<CChar>,
    _ voiceIdentifier: UnsafePointer<CChar>?,
    _ metadata: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOutput: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> UnsafeMutableRawPointer? {
    metadata.pointee = nil
    errorOutput.pointee = nil
    guard let locale = String(validatingUTF8: locale) else {
        errorOutput.pointee = strdup("locale is not valid UTF-8")
        return nil
    }
    let requestedVoice: String?
    if let voiceIdentifier {
        guard let decodedVoice = String(validatingUTF8: voiceIdentifier) else {
            errorOutput.pointee = strdup(
                "voice identifier is not valid UTF-8"
            )
            return nil
        }
        requestedVoice = decodedVoice
    } else {
        requestedVoice = nil
    }
    let voice: AVSpeechSynthesisVoice?
    if let requestedVoice, !requestedVoice.isEmpty {
        voice = AVSpeechSynthesisVoice(identifier: requestedVoice)
    } else {
        voice = AVSpeechSynthesisVoice(language: locale)
    }
    guard let voice else {
        errorOutput.pointee = strdup(
            "no installed voice for locale \(locale)"
        )
        return nil
    }
    guard voice.language == locale ||
        voice.language.split(separator: "-").first ==
          locale.split(separator: "-").first
    else {
        errorOutput.pointee = strdup(
            "voice \(voice.identifier) does not match locale \(locale)"
        )
        return nil
    }
    let session = TtsSession(requestedLocale: locale, voice: voice)
    guard writeJSON(session.metadata, to: metadata) == 0 else {
        errorOutput.pointee = metadata.pointee
        metadata.pointee = nil
        return nil
    }
    return Unmanaged.passRetained(session).toOpaque()
}

@_cdecl("cuttledoc_tts_session_synthesize")
public func cuttledoc_tts_session_synthesize(
    _ handle: UnsafeMutableRawPointer,
    _ text: UnsafePointer<CChar>,
    _ samples: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>,
    _ sampleCount: UnsafeMutablePointer<UInt64>,
    _ sampleRateHz: UnsafeMutablePointer<UInt32>,
    _ summary: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOutput: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>
) -> Int32 {
    samples.pointee = nil
    sampleCount.pointee = 0
    sampleRateHz.pointee = 0
    summary.pointee = nil
    errorOutput.pointee = nil
    guard let text = String(validatingUTF8: text), !text.isEmpty else {
        errorOutput.pointee = strdup("text must be non-empty UTF-8")
        return 2
    }
    switch ttsSession(handle).synthesize(text) {
    case let .success((audio, stats)):
        let byteCount = audio.samples.count * MemoryLayout<Float>.size
        guard let allocation = malloc(byteCount) else {
            errorOutput.pointee = strdup("could not allocate output audio")
            return 1
        }
        audio.samples.withUnsafeBytes {
            allocation.copyMemory(from: $0.baseAddress!, byteCount: byteCount)
        }
        samples.pointee = allocation.bindMemory(
            to: Float.self,
            capacity: audio.samples.count
        )
        sampleCount.pointee = UInt64(audio.samples.count)
        sampleRateHz.pointee = audio.sampleRateHz
        return writeJSON(stats, to: summary)
    case let .failure(error as CancellationError):
        _ = writeJSON(error.stats, to: summary)
        errorOutput.pointee = strdup("synthesis cancelled")
        return 3
    case let .failure(error as NSError) where error.code == 4:
        errorOutput.pointee = strdup(error.localizedDescription)
        return 4
    case let .failure(error):
        errorOutput.pointee = strdup(String(describing: error))
        return 1
    }
}

@_cdecl("cuttledoc_tts_session_cancel")
public func cuttledoc_tts_session_cancel(_ handle: UnsafeMutableRawPointer) {
    ttsSession(handle).cancel()
}

@_cdecl("cuttledoc_tts_session_destroy")
public func cuttledoc_tts_session_destroy(_ handle: UnsafeMutableRawPointer) {
    ttsSession(handle).close()
    Unmanaged<TtsSession>.fromOpaque(handle).release()
}

@_cdecl("cuttledoc_tts_free_audio")
public func cuttledoc_tts_free_audio(_ samples: UnsafeMutablePointer<Float>?) {
    free(samples)
}

@_cdecl("cuttledoc_tts_free_string")
public func cuttledoc_tts_free_string(
    _ value: UnsafeMutablePointer<CChar>?
) {
    free(value)
}
