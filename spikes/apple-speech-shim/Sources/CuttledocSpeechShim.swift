import AVFoundation
import Darwin
import Foundation
import Speech

private final class TranscriptionBox: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<String, Error>?

    func store(_ result: Result<String, Error>) {
        lock.lock()
        self.result = result
        lock.unlock()
    }

    func take() -> Result<String, Error>? {
        lock.lock()
        defer { lock.unlock() }
        return result
    }
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

    let completion = DispatchSemaphore(value: 0)
    let box = TranscriptionBox()
    Task {
        do {
            box.store(.success(try await transcribe(path: path)))
        } catch {
            box.store(.failure(error))
        }
        completion.signal()
    }
    completion.wait()

    switch box.take() {
    case let .success(text):
        output.pointee = strdup(text)
        return 0
    case let .failure(error):
        output.pointee = strdup(String(describing: error))
        return 1
    case nil:
        output.pointee = strdup("Swift task completed without a result")
        return 1
    }
}

@_cdecl("cuttledoc_speech_free_string")
public func cuttledoc_speech_free_string(_ value: UnsafeMutablePointer<CChar>?) {
    free(value)
}

private func transcribe(path: String) async throws -> String {
    guard let locale = await SpeechTranscriber.supportedLocale(
        equivalentTo: Locale(identifier: "en-US")
    ) else {
        throw NSError(
            domain: "CuttledocSpeechShim",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "en-US is not supported"]
        )
    }

    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        try await request.downloadAndInstall()
    }

    let audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    async let transcript: String = try transcriber.results.reduce("") { text, result in
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
