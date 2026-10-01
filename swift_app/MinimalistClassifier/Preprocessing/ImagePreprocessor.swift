import UIKit
import Accelerate

// MARK: - Feature Vector

/// Holds all 49 extracted features matching model_metadata.json order exactly.
/// Field names match the Core ML model input names verbatim.
struct FeatureVector {
    // -- Metadata (derived from resized image dimensions)
    var photo_width: Double      // always 256 after resize
    var photo_height: Double     // always 256 after resize
    var aspect_ratio: Double     // always 1.0 after resize
    var download_count: Double   // placeholder — 0 at inference time

    // -- Global (8)
    var brightness: Double
    var contrast: Double
    var sharpness: Double
    var edge_density: Double
    var saturation: Double
    var colorfulness: Double
    var texture_ent: Double
    var hue_std: Double

    // -- Grid per-cell (9 cells × 3 features, interleaved: edge, sharp, bright)
    var edge_1: Double;  var sharp_1: Double;  var bright_1: Double
    var edge_2: Double;  var sharp_2: Double;  var bright_2: Double
    var edge_3: Double;  var sharp_3: Double;  var bright_3: Double
    var edge_4: Double;  var sharp_4: Double;  var bright_4: Double
    var edge_5: Double;  var sharp_5: Double;  var bright_5: Double
    var edge_6: Double;  var sharp_6: Double;  var bright_6: Double
    var edge_7: Double;  var sharp_7: Double;  var bright_7: Double
    var edge_8: Double;  var sharp_8: Double;  var bright_8: Double
    var edge_9: Double;  var sharp_9: Double;  var bright_9: Double

    // -- Composition (10)
    var edge_max_ratio: Double
    var edge_std: Double
    var empty_ratio: Double
    var center_ratio: Double
    var sharp_max_ratio: Double
    var sharp_std: Double
    var thirds_ratio: Double
    var vertical_balance: Double
    var horiz_balance: Double
    var bright_std: Double
}

// MARK: - Preprocessor

/// Translates a UIImage into a FeatureVector following the pipeline in
/// `inference_preprocessing_pipeline.md` and `swift_inference_params.json`.
///
/// ⚠️  DO NOT modify formulas here. They must match the Python training pipeline exactly.
enum ImagePreprocessor {

    // MARK: Constants (from swift_inference_params.json)
    private static let IMG_SIZE   = 256
    private static let CELL_SIZE  = 85   // 256 // 3
    private static let CANNY_LOW  : UInt8 = 50
    private static let CANNY_HIGH : UInt8 = 150
    private static let LBP_P      = 8
    private static let LBP_BINS   = 10
    private static let LOW_EDGE   = 0.05

    // MARK: - Public Entry Point

    /// Extracts all 49 features from `image`. Returns nil if pixel data can't be read.
    static func extract(from image: UIImage) -> FeatureVector? {
        // ── STEP 1: Resize to 256×256 ──────────────────────────────────────
        guard let resized = image.resized(to: CGSize(width: IMG_SIZE, height: IMG_SIZE)) else { return nil }
        // Obtain raw RGBA bytes (uint8, row-major)
        guard let rgbaBytes = resized.rgbaPixels() else { return nil }

        let W = IMG_SIZE, H = IMG_SIZE

        // Build float RGB planes and uint8 planes (all needed downstream)
        // gray_float : float64 equivalent — [0,1]
        // gray_u8    : uint8 [0,255] — for Canny
        var grayFloat = [Float](repeating: 0, count: W * H)
        var grayU8    = [UInt8](repeating: 0, count: W * H)
        var hue       = [Float](repeating: 0, count: W * H)
        var sat       = [Float](repeating: 0, count: W * H)
        // val channel (not needed globally but computed in HSV loop)

        // R, G, B as Float in [0,255] — needed for colorfulness
        var rF = [Float](repeating: 0, count: W * H)
        var gF = [Float](repeating: 0, count: W * H)
        var bF = [Float](repeating: 0, count: W * H)

        for i in 0 ..< W * H {
            let r = Float(rgbaBytes[i * 4 + 0])
            let g = Float(rgbaBytes[i * 4 + 1])
            let b = Float(rgbaBytes[i * 4 + 2])
            rF[i] = r; gF[i] = g; bF[i] = b

            // Rec.709 grayscale  (skimage rgb2gray coefficients)
            let gray = 0.2125 * r + 0.7154 * g + 0.0721 * b   // float, 0-255 scaled
            let grayNorm = gray / 255.0                         // [0,1]
            grayFloat[i] = grayNorm
            grayU8[i]    = UInt8(min(255, max(0, Int(gray.rounded()))))

            // HSV conversion (standard algorithm, all channels [0,1])
            let rf = r / 255, gf = g / 255, bf = b / 255
            let cmax = max(rf, gf, bf), cmin = min(rf, gf, bf)
            let delta = cmax - cmin

            // Saturation
            sat[i] = cmax > 0 ? delta / cmax : 0

            // Hue
            var h: Float = 0
            if delta > 0 {
                if cmax == rf      { h = ((gf - bf) / delta).truncatingRemainder(dividingBy: 6) }
                else if cmax == gf { h = (bf - rf) / delta + 2 }
                else               { h = (rf - gf) / delta + 4 }
                h = h / 6
                if h < 0 { h += 1 }
            }
            hue[i] = h
        }

        // ── STEP 2: Global Features ─────────────────────────────────────────

        // Feature 1 — brightness = mean(gray_float)
        let brightness = Double(mean(grayFloat))

        // Feature 2 — contrast = std(gray_float)
        let contrast = Double(std(grayFloat))

        // Feature 3 — sharpness = var(laplace(gray_float))
        let laplaceGlobal = laplacian(grayFloat, width: W, height: H)
        let sharpness = Double(variance(laplaceGlobal))

        // Feature 4 — edge_density  (Canny on uint8 gray)
        let edgesGlobal = cannyEdges(grayU8, width: W, height: H,
                                     low: CANNY_LOW, high: CANNY_HIGH)
        // edges are 0 or 255; divide mean by 255
        let edgeDensity = Double(mean(edgesGlobal.map { Float($0) })) / 255.0

        // Feature 5 — saturation = mean(S channel)
        let saturation = Double(mean(sat))

        // Feature 6 — colorfulness (Hasler & Süsstrunk 2003)
        // R,G,B are float [0,255] — NOT divided by 255
        let rg = zip(rF, gF).map { $0 - $1 }        // R - G
        let yb = zip(zip(rF, gF).map { ($0 + $1) * 0.5 }, bF).map { $0 - $1 }  // 0.5(R+G) - B
        let colorfulness = Double(
            sqrt(variance(rg) + variance(yb)) +
            0.3 * sqrt(mean(rg) * mean(rg) + mean(yb) * mean(yb))
        )

        // Feature 7 — texture_ent (LBP entropy, P=8 R=1 uniform, 10 bins)
        let textureEnt = lbpEntropy(grayFloat, width: W, height: H)

        // Feature 8 — hue_std
        let hueStd = Double(std(hue))

        // ── STEP 3: Grid Features (3×3, cells are 85×85) ───────────────────
        // Grid features are stored interleaved per cell in the JSON:
        // edge_1, sharp_1, bright_1, edge_2, sharp_2, bright_2, ...

        var cellEdge   = [Double](repeating: 0, count: 9)
        var cellSharp  = [Double](repeating: 0, count: 9)
        var cellBright = [Double](repeating: 0, count: 9)

        for row in 0 ..< 3 {
            for col in 0 ..< 3 {
                let idx = row * 3 + col   // 0-based index

                // Extract cell pixels (float gray and uint8 gray)
                var cellGrayF = [Float](repeating: 0, count: CELL_SIZE * CELL_SIZE)
                var cellGrayU = [UInt8](repeating: 0, count: CELL_SIZE * CELL_SIZE)

                for r in 0 ..< CELL_SIZE {
                    for c in 0 ..< CELL_SIZE {
                        let srcRow = row * CELL_SIZE + r
                        let srcCol = col * CELL_SIZE + c
                        let srcIdx = srcRow * W + srcCol
                        let dstIdx = r * CELL_SIZE + c
                        cellGrayF[dstIdx] = grayFloat[srcIdx]
                        cellGrayU[dstIdx] = grayU8[srcIdx]
                    }
                }

                // edge_{idx+1}
                let cellEdges = cannyEdges(cellGrayU, width: CELL_SIZE, height: CELL_SIZE,
                                           low: CANNY_LOW, high: CANNY_HIGH)
                cellEdge[idx] = Double(mean(cellEdges.map { Float($0) })) / 255.0

                // sharp_{idx+1}
                let cellLap = laplacian(cellGrayF, width: CELL_SIZE, height: CELL_SIZE)
                cellSharp[idx] = Double(variance(cellLap))

                // bright_{idx+1}
                cellBright[idx] = Double(mean(cellGrayF))
            }
        }

        // ── STEP 4: Composition Features ───────────────────────────────────
        let edges  = cellEdge
        let sharps = cellSharp
        let brights = cellBright

        // edge_max_ratio
        let eSum = edges.reduce(0, +)
        let edgeMaxRatio = edges.max()! / (eSum + 1e-8)

        // edge_std
        let edgeStdVal = stdD(edges)

        // empty_ratio
        let emptyRatio = Double(edges.filter { $0 < LOW_EDGE }.count) / 9.0

        // center_ratio  (center cell is index 4, i.e. edge_5 in 1-based)
        let centerEdge = edges[4]
        let othersEdge = edges.filter { $0 != centerEdge }
        let othersMean = othersEdge.reduce(0, +) / Double(othersEdge.count) + 1e-8
        let centerRatio = centerEdge / othersMean

        // sharp_max_ratio
        let sSum = sharps.reduce(0, +)
        let sharpMaxRatio = sharps.max()! / (sSum + 1e-8)

        // sharp_std
        let sharpStdVal = stdD(sharps)

        // thirds_ratio  (1-based cells 2,4,6,8 → 0-based 1,3,5,7)
        let thirdsEdges = [edges[1], edges[3], edges[5], edges[7]]
        let thirdsMean  = thirdsEdges.reduce(0, +) / 4.0
        let allMean     = eSum / 9.0
        let thirdsRatio = thirdsMean / (allMean + 1e-8)

        // vertical_balance  (top row 0,1,2 vs bottom row 6,7,8)
        let topMean    = (edges[0] + edges[1] + edges[2]) / 3.0
        let bottomMean = (edges[6] + edges[7] + edges[8]) / 3.0
        let vertBalance = topMean / (bottomMean + 1e-8)

        // horiz_balance  (left col 0,3,6 vs right col 2,5,8)
        let leftMean  = (edges[0] + edges[3] + edges[6]) / 3.0
        let rightMean = (edges[2] + edges[5] + edges[8]) / 3.0
        let horizBalance = leftMean / (rightMean + 1e-8)

        // bright_std
        let brightStdVal = stdD(brights)

        // ── STEP 5: Assemble FeatureVector (JSON order) ────────────────────
        return FeatureVector(
            // Metadata
            photo_width:    Double(W),
            photo_height:   Double(H),
            aspect_ratio:   1.0,
            download_count: 0.0,
            // Global
            brightness:   brightness,
            contrast:     contrast,
            sharpness:    sharpness,
            edge_density: edgeDensity,
            saturation:   saturation,
            colorfulness: colorfulness,
            texture_ent:  textureEnt,
            hue_std:      hueStd,
            // Grid — interleaved per cell
            edge_1: cellEdge[0],  sharp_1: cellSharp[0],  bright_1: cellBright[0],
            edge_2: cellEdge[1],  sharp_2: cellSharp[1],  bright_2: cellBright[1],
            edge_3: cellEdge[2],  sharp_3: cellSharp[2],  bright_3: cellBright[2],
            edge_4: cellEdge[3],  sharp_4: cellSharp[3],  bright_4: cellBright[3],
            edge_5: cellEdge[4],  sharp_5: cellSharp[4],  bright_5: cellBright[4],
            edge_6: cellEdge[5],  sharp_6: cellSharp[5],  bright_6: cellBright[5],
            edge_7: cellEdge[6],  sharp_7: cellSharp[6],  bright_7: cellBright[6],
            edge_8: cellEdge[7],  sharp_8: cellSharp[7],  bright_8: cellBright[7],
            edge_9: cellEdge[8],  sharp_9: cellSharp[8],  bright_9: cellBright[8],
            // Composition
            edge_max_ratio:   edgeMaxRatio,
            edge_std:         edgeStdVal,
            empty_ratio:      emptyRatio,
            center_ratio:     centerRatio,
            sharp_max_ratio:  sharpMaxRatio,
            sharp_std:        sharpStdVal,
            thirds_ratio:     thirdsRatio,
            vertical_balance: vertBalance,
            horiz_balance:    horizBalance,
            bright_std:       brightStdVal
        )
    }
}

// MARK: - Math Helpers (Float)

private func mean(_ a: [Float]) -> Float {
    guard !a.isEmpty else { return 0 }
    var result: Float = 0
    vDSP_meanv(a, 1, &result, vDSP_Length(a.count))
    return result
}

private func variance(_ a: [Float]) -> Float {
    guard a.count > 1 else { return 0 }
    let m = mean(a)
    var diffs = a.map { ($0 - m) * ($0 - m) }
    var result: Float = 0
    vDSP_meanv(diffs, 1, &result, vDSP_Length(diffs.count))
    return result
}

private func std(_ a: [Float]) -> Float { sqrt(variance(a)) }

// MARK: - Math Helpers (Double)

private func stdD(_ a: [Double]) -> Double {
    guard a.count > 1 else { return 0 }
    let m = a.reduce(0, +) / Double(a.count)
    let v = a.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(a.count)
    return sqrt(v)
}

// MARK: - Laplacian

/// Applies a 3×3 discrete Laplacian kernel (same as skimage.filters.laplace default).
///
/// Kernel:
///   0  1  0
///   1 -4  1
///   0  1  0
///
/// Edges are zero-padded.
private func laplacian(_ input: [Float], width: Int, height: Int) -> [Float] {
    var output = [Float](repeating: 0, count: width * height)
    for y in 1 ..< height - 1 {
        for x in 1 ..< width - 1 {
            let c  = input[y * width + x]
            let up = input[(y - 1) * width + x]
            let dn = input[(y + 1) * width + x]
            let lt = input[y * width + (x - 1)]
            let rt = input[y * width + (x + 1)]
            output[y * width + x] = up + dn + lt + rt - 4 * c
        }
    }
    return output
}

// MARK: - Canny Edge Detection

/// Simplified Canny edge detector operating on uint8 grayscale input.
///
/// Pipeline:
///   1. Gaussian blur (3×3 approximation)
///   2. Sobel gradient magnitude
///   3. Non-maximum suppression (simplified 4-direction)
///   4. Double threshold + hysteresis
///
/// ⚠️  Input must be uint8 [0,255]. Output is uint8: 0 or 255.
private func cannyEdges(
    _ input: [UInt8],
    width: Int,
    height: Int,
    low: UInt8,
    high: UInt8
) -> [UInt8] {

    let n = width * height
    // 1. Convert to float and apply a simple 3×3 Gaussian blur
    var f = input.map { Float($0) }
    var blurred = [Float](repeating: 0, count: n)

    // 3×3 Gaussian kernel weights (approximation):  1 2 1 / 2 4 2 / 1 2 1  /16
    let kw: [[Float]] = [[1,2,1],[2,4,2],[1,2,1]]
    for y in 1 ..< height - 1 {
        for x in 1 ..< width - 1 {
            var val: Float = 0
            for ky in -1...1 {
                for kx in -1...1 {
                    val += kw[ky + 1][kx + 1] * f[(y + ky) * width + (x + kx)]
                }
            }
            blurred[y * width + x] = val / 16.0
        }
    }

    // 2. Sobel gradients
    var gx = [Float](repeating: 0, count: n)
    var gy = [Float](repeating: 0, count: n)
    var mag = [Float](repeating: 0, count: n)
    var ang = [Float](repeating: 0, count: n)  // angle in degrees 0,45,90,135

    for y in 1 ..< height - 1 {
        for x in 1 ..< width - 1 {
            let idx = y * width + x
            // Sobel X
            gx[idx] = (blurred[(y-1)*width+(x+1)] + 2*blurred[y*width+(x+1)] + blurred[(y+1)*width+(x+1)])
                    - (blurred[(y-1)*width+(x-1)] + 2*blurred[y*width+(x-1)] + blurred[(y+1)*width+(x-1)])
            // Sobel Y
            gy[idx] = (blurred[(y+1)*width+(x-1)] + 2*blurred[(y+1)*width+x] + blurred[(y+1)*width+(x+1)])
                    - (blurred[(y-1)*width+(x-1)] + 2*blurred[(y-1)*width+x] + blurred[(y-1)*width+(x+1)])
            mag[idx] = sqrt(gx[idx]*gx[idx] + gy[idx]*gy[idx])
            // Quantise angle to 0/45/90/135 degrees
            var a = atan2(gy[idx], gx[idx]) * 180 / .pi
            if a < 0 { a += 180 }
            if      (a < 22.5 || a >= 157.5) { ang[idx] = 0 }
            else if (a < 67.5)               { ang[idx] = 45 }
            else if (a < 112.5)              { ang[idx] = 90 }
            else                             { ang[idx] = 135 }
        }
    }

    // 3. Non-maximum suppression
    var nms = [Float](repeating: 0, count: n)
    for y in 1 ..< height - 1 {
        for x in 1 ..< width - 1 {
            let idx = y * width + x
            let m = mag[idx]
            var q: Float = 0, r: Float = 0
            switch ang[idx] {
            case 0:
                q = mag[y * width + (x+1)]; r = mag[y * width + (x-1)]
            case 45:
                q = mag[(y-1)*width+(x+1)]; r = mag[(y+1)*width+(x-1)]
            case 90:
                q = mag[(y-1)*width+x];     r = mag[(y+1)*width+x]
            default: // 135
                q = mag[(y-1)*width+(x-1)]; r = mag[(y+1)*width+(x+1)]
            }
            nms[idx] = (m >= q && m >= r) ? m : 0
        }
    }

    // 4. Double threshold + hysteresis
    let loF = Float(low), hiF = Float(high)
    var edges = [UInt8](repeating: 0, count: n)

    // Mark strong / weak edges
    for i in 0 ..< n {
        if nms[i] >= hiF      { edges[i] = 255 }   // strong
        else if nms[i] >= loF { edges[i] = 128 }   // weak (candidate)
    }

    // Hysteresis: promote weak edges connected to strong edges
    for y in 1 ..< height - 1 {
        for x in 1 ..< width - 1 {
            let idx = y * width + x
            guard edges[idx] == 128 else { continue }
            let neighbours = [
                edges[(y-1)*width+(x-1)], edges[(y-1)*width+x], edges[(y-1)*width+(x+1)],
                edges[y*width+(x-1)],                            edges[y*width+(x+1)],
                edges[(y+1)*width+(x-1)], edges[(y+1)*width+x], edges[(y+1)*width+(x+1)]
            ]
            edges[idx] = neighbours.contains(255) ? 255 : 0
        }
    }

    return edges
}

// MARK: - LBP Entropy

/// Local Binary Pattern entropy.
/// P=8 neighbours, R=1.0 (integer pixel offsets), uniform method, 10-bin histogram, density=True, log2.
///
/// Uniform LBP has (P*(P-1)+2) = 58 uniform patterns + 1 non-uniform → 59 labels total.
/// The histogram is then computed into `lbp_bins=10` bins over [0, 59].
private func lbpEntropy(_ gray: [Float], width: Int, height: Int) -> Double {
    // Neighbour offsets for P=8, R=1 (clockwise from right, matching skimage convention)
    let offsets: [(Int, Int)] = [(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)]

    var lbpValues = [Int]()
    lbpValues.reserveCapacity(width * height)

    for y in 0 ..< height {
        for x in 0 ..< width {
            let center = gray[y * width + x]
            var pattern = 0
            for (i, (dx, dy)) in offsets.enumerated() {
                let nx = x + dx, ny = y + dy
                // Boundary: replicate edge
                let nx2 = min(max(nx, 0), width - 1)
                let ny2 = min(max(ny, 0), height - 1)
                let neighbour = gray[ny2 * width + nx2]
                if neighbour >= center { pattern |= (1 << i) }
            }
            lbpValues.append(uniformLBP(pattern: pattern, P: 8))
        }
    }

    // Build 10-bin histogram over range [0, maxUniform] where maxUniform = P*(P-1)+2 = 58
    // density=True: normalise so the histogram integrates to 1
    let maxLabel = 58  // P*(P-1)+2  (0-based, so 59 possible labels: 0…58)
    var hist = [Double](repeating: 0, count: 10)
    let binWidth = Double(maxLabel + 1) / 10.0  // ~5.9 per bin

    for v in lbpValues {
        let bin = min(9, Int(Double(v) / binWidth))
        hist[bin] += 1
    }
    // Normalise (density=True equivalent: divide by total × binWidth)
    let total = Double(lbpValues.count) * binWidth
    hist = hist.map { $0 / total }

    // Shannon entropy (log2), non-zero bins only
    let entropy = hist.filter { $0 > 0 }.reduce(0.0) { acc, p in
        acc - p * log2(p)
    }
    return entropy
}

/// Maps an 8-bit LBP binary pattern to its uniform label [0…58].
/// Uniform = at most 2 bitwise transitions in the circular pattern.
/// Non-uniform patterns all map to label 58.
private func uniformLBP(pattern: Int, P: Int) -> Int {
    // Count transitions (0→1 or 1→0) in circular bit string
    var transitions = 0
    for i in 0 ..< P {
        let bit1 = (pattern >> i) & 1
        let bit2 = (pattern >> ((i + 1) % P)) & 1
        if bit1 != bit2 { transitions += 1 }
    }
    guard transitions <= 2 else { return P * (P - 1) + 2 }  // non-uniform label
    // Uniform label = popcount (number of set bits)
    return pattern.nonzeroBitCount
}

// MARK: - UIImage helpers

extension UIImage {
    /// Resizes the image to `size` using UIGraphicsImageRenderer.
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsImageRenderer(size: size).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }

    /// Returns raw RGBA bytes (UInt8), row-major, 4 bytes per pixel.
    func rgbaPixels() -> [UInt8]? {
        let width = Int(size.width), height = Int(size.height)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let ctx = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        guard let cg = cgImage else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixels
    }
}
