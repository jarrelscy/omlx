// Inkling (oMLX) desktop widget: a small window on the desktop that shows the oMLX server's live status
// (GET /api/status) every 2 seconds, with live decode and prefill speed graphs. Build with build.sh.
//
// Settings (defaults write ai.omlx.inkling-widget <key> <value>):
//   url     status URL (default http://127.0.0.1:8000/api/status)
//   apiKey  API key if the server requires one
// Right-click the widget for "Keep on top", "Open oMLX admin in browser" and "Quit". Drag it to move it.

import AppKit
import IOKit
import IOKit.ps

let defaults = UserDefaults.standard      // the app's own domain, ai.omlx.inkling-widget

// MARK: - system

/// The GPU's busy percentage, as Activity Monitor shows it.
func gpuPercent() -> Int? {
    var iterator: io_iterator_t = 0
    guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS else { return nil }
    defer { IOObjectRelease(iterator) }
    var service = IOIteratorNext(iterator)
    while service != 0 {
        defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }
        if let stats = IORegistryEntryCreateCFProperty(service, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? [String: Any], let busy = stats["Device Utilization %"] as? Int { return busy }
    }
    return nil
}

/// Memory macOS can hand out now (free, inactive and purgeable pages), in GiB.
func availableGiB() -> Double? {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    guard result == KERN_SUCCESS else { return nil }
    let pages = Double(stats.free_count) + Double(stats.inactive_count) + Double(stats.purgeable_count)
    return pages * Double(vm_kernel_page_size) / 1_073_741_824
}

/// What the model folder says about the Inkling model: its loop passes and whether the drafter is on. `folder` is
/// the model's own folder when the server reports it (mkserve, which always drafts), else oMLX's model folder.
func modelNotes(_ model: String, folder given: String? = nil) -> String {
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    var notes: [String] = []
    let folder = given ?? "\(home)/.omlx/models/\(model)"
    if given != nil { notes.append((folder as NSString).lastPathComponent) }
    if let link = try? FileManager.default.destinationOfSymbolicLink(atPath: folder) { notes.append((link as NSString).lastPathComponent) }
    if let data = FileManager.default.contents(atPath: folder + "/config.json"),
       let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let passes = num(dict(dict(config["text_config"])["loop_config"])["num_passes"]) {
        notes.append("\(Int(passes)) loops")
    }
    if given != nil {
        notes.append(FileManager.default.fileExists(atPath: folder + "/mtp.safetensors") ? "drafter on" : "drafter off")
    } else if let data = FileManager.default.contents(atPath: "\(home)/.omlx/model_settings.json"),
       let settings = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
        let on = (dict(dict(settings["models"])[model])["mtp_enabled"] as? Bool) ?? false
        notes.append("drafter " + (on ? "on" : "off"))
    }
    return notes.joined(separator: " · ")
}

// MARK: - formatting

func compact(_ value: Double?) -> String {
    guard let v = value else { return "–" }
    if v >= 1_000_000 { return String(format: "%.2fM", v / 1_000_000) }
    if v >= 10_000 { return String(format: "%.0fK", v / 1_000) }
    if v >= 1_000 { return String(format: "%.1fK", v / 1_000) }
    return String(format: "%.0f", v)
}

func rate(_ value: Double?) -> String {
    guard let v = value else { return "–" }
    return String(format: "%.1f", v)
}

func duration(_ seconds: Double?) -> String {
    let s = Int(seconds ?? 0)
    if s >= 3600 { return "\(s / 3600)h\(String(format: "%02d", s % 3600 / 60))m" }
    if s >= 60 { return "\(s / 60)m\(String(format: "%02d", s % 60))s" }
    return "\(s)s"
}

func num(_ any: Any?) -> Double? {
    if let n = any as? NSNumber { return n.doubleValue }
    return nil
}

func dict(_ any: Any?) -> [String: Any] { any as? [String: Any] ?? [:] }

// MARK: - battery

func batteryText() -> String? {
    guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
          let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] else { return nil }
    for source in list {
        guard let d = IOPSGetPowerSourceDescription(info, source)?.takeUnretainedValue() as? [String: Any],
              let capacity = d[kIOPSCurrentCapacityKey] as? Int else { continue }
        let onAC = (d[kIOPSPowerSourceStateKey] as? String) == kIOPSACPowerValue
        let charging = d[kIOPSIsChargingKey] as? Bool ?? false
        return "battery \(capacity)%" + (charging ? " ⚡" : (onAC ? " (AC)" : " (on battery)"))
    }
    return nil
}

// MARK: - speed graph

/// Live decode and prefill speed over the last three minutes, one sample per poll, as two small charts.
final class SpeedGraph: NSView {
    struct Sample { var decode: Double?; var prefill: Double? }
    private(set) var samples: [Sample] = []
    let capacity = 90                      // 3 minutes at one sample every 2 seconds
    static let height: CGFloat = 112

    override var isFlipped: Bool { true }

    func push(decode: Double?, prefill: Double?) {
        samples.append(Sample(decode: decode.flatMap { $0 > 0 ? $0 : nil }, prefill: prefill.flatMap { $0 > 0 ? $0 : nil }))
        if samples.count > capacity { samples.removeFirst(samples.count - capacity) }
        needsDisplay = true
    }

    /// Fills in a speed for a request that finished between polls, over the polls the work spanned. The work
    /// ended `endingAgo` seconds before the latest sample. Samples that already hold a speed are kept.
    func backfill(_ which: WritableKeyPath<Sample, Double?>, _ value: Double?, seconds: Double, endingAgo: Double = 0,
                  interval: Double = 2) {
        guard let value, value > 0, !samples.isEmpty else { return }
        let end = samples.count - 1 - max(0, Int((endingAgo / interval).rounded()))
        guard end >= 0 else { return }
        let start = max(0, end - max(1, Int((seconds / interval).rounded(.up))) + 1)
        for index in start...end where samples[index][keyPath: which] == nil { samples[index][keyPath: which] = value }
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        let half = (bounds.height - 8) / 2
        panel(NSRect(x: 0, y: 0, width: bounds.width, height: half), title: "DECODE", values: samples.map(\.decode),
              color: .systemGreen)
        panel(NSRect(x: 0, y: half + 8, width: bounds.width, height: half), title: "PREFILL", values: samples.map(\.prefill),
              color: .systemCyan)
    }

    private func panel(_ rect: NSRect, title: String, values: [Double?], color: NSColor) {
        let grey = NSColor(white: 0.62, alpha: 1)
        let head = NSFont.systemFont(ofSize: 10, weight: .semibold)
        let mono = NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular)
        let peak = values.compactMap { $0 }.max() ?? 0
        let current = values.last ?? nil
        (title + " tok/s" as NSString).draw(at: NSPoint(x: rect.minX, y: rect.minY), withAttributes: [.font: head, .foregroundColor: grey])
        let right = "now \(current.map(rate) ?? "–") · peak \(peak > 0 ? rate(peak) : "–")" as NSString
        let attributes: [NSAttributedString.Key: Any] = [.font: mono, .foregroundColor: current == nil ? grey : color]
        right.draw(at: NSPoint(x: rect.maxX - right.size(withAttributes: attributes).width, y: rect.minY), withAttributes: attributes)

        let plot = NSRect(x: rect.minX, y: rect.minY + 15, width: rect.width, height: rect.height - 15)
        NSColor(white: 1, alpha: 0.06).setFill()
        NSBezierPath(roundedRect: plot, xRadius: 4, yRadius: 4).fill()
        guard peak > 0 else { return }
        let top = peak * 1.15
        let step = plot.width / CGFloat(capacity - 1)
        let offset = capacity - values.count
        func point(_ index: Int, _ value: Double) -> NSPoint {
            NSPoint(x: plot.minX + CGFloat(index + offset) * step, y: plot.maxY - CGFloat(value / top) * plot.height)
        }
        // Runs of consecutive samples are drawn as filled lines; a lone sample as a dot.
        var run: [NSPoint] = []
        func flush() {
            defer { run = [] }
            guard let first = run.first, let last = run.last else { return }
            if run.count == 1 {
                color.setFill()
                NSBezierPath(ovalIn: NSRect(x: first.x - 2, y: first.y - 2, width: 4, height: 4)).fill()
                return
            }
            let area = NSBezierPath()
            area.move(to: NSPoint(x: first.x, y: plot.maxY))
            run.forEach { area.line(to: $0) }
            area.line(to: NSPoint(x: last.x, y: plot.maxY))
            area.close()
            color.withAlphaComponent(0.18).setFill()
            area.fill()
            let line = NSBezierPath()
            line.move(to: first)
            run.dropFirst().forEach { line.line(to: $0) }
            line.lineWidth = 1.5
            line.lineJoinStyle = .round
            color.setStroke()
            line.stroke()
        }
        for (index, value) in values.enumerated() {
            if let value { run.append(point(index, value)) } else { flush() }
        }
        flush()
    }
}

// MARK: - view

final class Widget: NSObject, NSApplicationDelegate {
    var panel: NSPanel!
    let label = NSTextField(labelWithString: "")
    let graph = SpeedGraph()
    var timer: Timer?

    let mono = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
    let monoBold = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
    let head = NSFont.systemFont(ofSize: 10, weight: .semibold)

    func applicationDidFinishLaunching(_ notification: Notification) {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 480, height: 250),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        panel.hidesOnDeactivate = false
        applyLevel()

        let effect = NSVisualEffectView()
        effect.material = .hudWindow
        effect.state = .active
        effect.blendingMode = .behindWindow
        effect.wantsLayer = true
        effect.layer?.cornerRadius = 14
        effect.layer?.masksToBounds = true
        panel.contentView = effect

        label.maximumNumberOfLines = 0
        label.lineBreakMode = .byWordWrapping
        label.translatesAutoresizingMaskIntoConstraints = false
        label.preferredMaxLayoutWidth = 450
        effect.addSubview(label)
        graph.translatesAutoresizingMaskIntoConstraints = false
        effect.addSubview(graph)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: effect.leadingAnchor, constant: 14),
            label.trailingAnchor.constraint(equalTo: effect.trailingAnchor, constant: -14),
            label.topAnchor.constraint(equalTo: effect.topAnchor, constant: 12),
            graph.leadingAnchor.constraint(equalTo: effect.leadingAnchor, constant: 14),
            graph.trailingAnchor.constraint(equalTo: effect.trailingAnchor, constant: -14),
            graph.topAnchor.constraint(equalTo: label.bottomAnchor, constant: 10),
            graph.heightAnchor.constraint(equalToConstant: SpeedGraph.height),
            graph.bottomAnchor.constraint(equalTo: effect.bottomAnchor, constant: -12),
        ])

        let menu = NSMenu()
        let top = NSMenuItem(title: "Keep on top", action: #selector(toggleTop), keyEquivalent: "")
        top.target = self
        top.state = defaults.bool(forKey: "keepOnTop") ? .on : .off
        menu.addItem(top)
        let open = NSMenuItem(title: "Open oMLX admin in browser", action: #selector(openMetrics), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Inkling Widget", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "")
        menu.addItem(quit)
        effect.menu = menu

        panel.setFrameAutosaveName("InklingWidgetFrame")
        if !panel.setFrameUsingName("InklingWidgetFrame"), let screen = NSScreen.main {
            let f = screen.visibleFrame
            panel.setFrameOrigin(NSPoint(x: f.maxX - 500, y: f.maxY - 560))
        }
        render(nil, error: "connecting…")
        if let index = CommandLine.arguments.firstIndex(of: "--snapshot"), index + 1 < CommandLine.arguments.count {
            snapshot(to: CommandLine.arguments[index + 1])
            return
        }
        panel.orderFrontRegardless()

        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in self?.poll() }
        poll()
    }

    /// Render one update into a PNG and quit (for checking the layout without a screen capture).
    func snapshot(to path: String) {
        var request = URLRequest(url: metricsURL, timeoutInterval: 4)
        if let key = defaults.string(forKey: "apiKey"), !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
            let object = data.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
            DispatchQueue.main.async {
                guard let self = self, let view = self.panel.contentView else { exit(1) }
                self.render(object, error: error?.localizedDescription ?? (object == nil ? "no data" : nil))
                view.layoutSubtreeIfNeeded()
                view.appearance = NSAppearance(named: .darkAqua)
                if let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                    view.cacheDisplay(in: view.bounds, to: rep)
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                }
                exit(0)
            }
        }.resume()
    }

    func applyLevel() {
        if defaults.bool(forKey: "keepOnTop") {
            panel.level = .floating
        } else {
            // Just above the desktop icons, below normal windows, like a desktop widget.
            panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.desktopIconWindow)) + 1)
        }
    }

    @objc func toggleTop(_ sender: NSMenuItem) {
        let on = !defaults.bool(forKey: "keepOnTop")
        defaults.set(on, forKey: "keepOnTop")
        sender.state = on ? .on : .off
        applyLevel()
        panel.orderFrontRegardless()
    }

    var metricsURL: URL {
        URL(string: defaults.string(forKey: "url") ?? "http://127.0.0.1:8000/api/status")!
    }

    @objc func openMetrics() { NSWorkspace.shared.open(URL(string: "/admin", relativeTo: metricsURL)!.absoluteURL) }

    func poll() {
        var request = URLRequest(url: metricsURL, timeoutInterval: 4)
        if let key = defaults.string(forKey: "apiKey"), !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
            let object = data.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
            DispatchQueue.main.async {
                if let m = object { self?.render(m, error: nil) }
                else { self?.render(nil, error: error?.localizedDescription ?? "no data") }
            }
        }.resume()
    }

    // MARK: rendering

    func render(_ m: [String: Any]?, error: String?) {
        let text = NSMutableAttributedString()
        let white = NSColor.white, grey = NSColor(white: 0.62, alpha: 1)

        func add(_ s: String, _ font: NSFont, _ color: NSColor) {
            text.append(NSAttributedString(string: s, attributes: [.font: font, .foregroundColor: color]))
        }
        func row(_ title: String, _ value: String, color: NSColor = .white) {
            add(title + "\t", head, grey)
            add(value + "\n", mono, color)
        }

        guard let m = m else {
            add("● ", monoBold, .systemRed)
            add("Inkling (oMLX)\n", NSFont.systemFont(ofSize: 13, weight: .semibold), white)
            add("not reachable: \(error ?? "")\n", mono, grey)
            add("start it with ~/startinkling.sh\n", mono, grey)
            if let b = batteryText() { add(b, mono, grey) }
            label.attributedStringValue = text
            fit()
            return
        }

        let active = num(m["active_requests"]) ?? 0, waiting = num(m["waiting_requests"]) ?? 0
        let loading = num(m["models_loading"]) ?? 0
        let models = (m["loaded_models"] as? [String]) ?? []
        let generated = num(m["total_completion_tokens"]) ?? 0
        // Tokens written since the last poll, for a live speed while a request runs.
        let now = Date()
        var live: Double?
        if let previous = lastTotals, active > 0, generated > previous.tokens {
            live = (generated - previous.tokens) / now.timeIntervalSince(previous.time)
        }
        lastTotals = (generated, now)
        // Live speeds from the requests running now, then the speeds of a request that finished between polls.
        let liveRows = dict(m["live"])
        func speeds(_ key: String) -> Double? {
            let values = ((liveRows[key] as? [[String: Any]]) ?? []).compactMap { num($0["tokens_per_second"]) }.filter { $0 > 0 }
            return values.isEmpty ? nil : values.reduce(0, +)
        }
        graph.push(decode: speeds("generating") ?? (m["live"] == nil ? live : nil), prefill: speeds("prefilling"))
        if let finished = m["last_request"] as? [String: Any], let at = num(finished["completed_at"]), at != lastFinished {
            if lastFinished != nil {
                let decodeSeconds = (num(finished["completion_tokens"]) ?? 0) / max(num(finished["decode_tps"]) ?? 1, 1)
                let ago = max(0, Date().timeIntervalSince1970 - at)
                graph.backfill(\.decode, num(finished["decode_tps"]), seconds: decodeSeconds, endingAgo: ago)
                graph.backfill(\.prefill, num(finished["prefill_tps"]), seconds: num(finished["prefill_seconds"]) ?? 0,
                               endingAgo: ago + decodeSeconds)
            }
            lastFinished = at
        } else if m["last_request"] is NSNull {
            lastFinished = lastFinished ?? 0     // no request yet: the first one to finish is filled in
        }
        if active > 0 { busySince = busySince ?? now } else { busySince = nil }

        add("● ", monoBold, active > 0 || loading > 0 ? .systemOrange : (models.isEmpty ? .systemGray : .systemGreen))
        add("Inkling  ", NSFont.systemFont(ofSize: 13, weight: .semibold), white)
        let mk = (m["server"] as? String) == "mkserve"
        add((mk ? "megakernel \(m["engine"] as? String ?? "")" : "oMLX \(m["version"] as? String ?? "")")
            + " · up \(duration(num(m["uptime_seconds"])))\n", mono, grey)

        let model = (m["default_model"] as? String) ?? models.first ?? ""
        row("MODEL", models.isEmpty ? (loading > 0 ? "loading…" : "none loaded") : "\(models.joined(separator: ", "))"
            + (model.isEmpty ? "" : "\n\t\(modelNotes(model, folder: (m["model_dir"] as? String).flatMap { $0.isEmpty ? nil : $0 }))"), color: models.isEmpty ? .systemYellow : white)
        row("REQUESTS", "\(Int(active)) processing · \(Int(waiting)) queued · \(Int(num(m["total_requests"]) ?? 0)) served",
            color: waiting > 0 ? .systemYellow : white)
        if active > 0 {
            let since = busySince.map { duration(now.timeIntervalSince($0)) } ?? ""
            row("NOW", live.map { "writing @ \(rate($0)) tok/s · \(since)" } ?? "working · \(since)", color: .systemOrange)
        } else {
            row("NOW", "idle", color: grey)
        }
        row("DECODE", "avg \(rate(num(m["avg_generation_tps"]))) tok/s this session")
        row("PREFILL", "avg \(rate(num(m["avg_prefill_tps"]))) tok/s this session")
        row("SESSION", "\(compact(generated)) generated · \(compact(num(m["total_prompt_tokens"]))) prompt"
            + " · \(rate(num(m["cache_efficiency"])))% from cache")
        row("MEMORY", (mk ? "context \(compact(num(m["max_context"]))) tokens"
                           : "models \(m["model_memory_used_formatted"] as? String ?? "–") of \(m["model_memory_max_formatted"] as? String ?? "–")")
            + " · free \(rate(availableGiB())) GiB")
        var system = "GPU \(gpuPercent().map(String.init) ?? "–")%"
        if let b = batteryText() { system += " · " + b }
        row("SYSTEM", system)

        // drop the final newline; one tab stop aligns the values, and wrapped lines indent to it
        if text.string.hasSuffix("\n") { text.deleteCharacters(in: NSRange(location: text.length - 1, length: 1)) }
        let style = NSMutableParagraphStyle()
        style.tabStops = [NSTextTab(textAlignment: .left, location: 78)]
        style.headIndent = 78
        style.lineSpacing = 2
        text.addAttribute(.paragraphStyle, value: style, range: NSRange(location: 0, length: text.length))
        label.attributedStringValue = text
        fit()
    }

    var lastTotals: (tokens: Double, time: Date)?
    var busySince: Date?
    var lastFinished: Double?

    func fit() {
        let size = label.fittingSize
        let frame = panel.frame
        let height = size.height + 24 + 10 + SpeedGraph.height
        if abs(frame.height - height) > 1 {
            panel.setFrame(NSRect(x: frame.minX, y: frame.maxY - height, width: 480, height: height), display: true)
        }
    }
}

let app = NSApplication.shared
let delegate = Widget()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
