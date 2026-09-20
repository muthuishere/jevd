// What this GPU can actually do, measured — not a spec sheet.
//
// Exists because docs/adr/0017 needed a denominator. "We sustain 4.8 TFLOP/s" says
// nothing until you know whether the ceiling is 5 or 50, and the shape that matters is
// not the big square matmul every vendor quotes: it is M=32, the number of tokens in an
// NLI pair. Apple's own MPS kernels at that shape are the fairest upper bound on what
// ggml could ever reach, because they are written by the people who made the silicon.
//
//   swiftc -O scripts/metal-peak.swift -o /tmp/metal-peak && /tmp/metal-peak
//
// Reports fp16 matmul TFLOP/s at several shapes and blit-copy bandwidth, best of 5.

import Metal
import MetalPerformanceShaders
import Foundation

// Measures achievable fp16 matmul throughput and memory bandwidth on this GPU.
// Not a spec sheet: MPSMatrixMultiplication on real buffers, best-of-N.

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
print("device: \(dev.name)")

func gemm(_ m: Int, _ k: Int, _ n: Int, iters: Int) -> Double {
    let rd = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k*2, dataType: .float16)
    let rdb = MPSMatrixDescriptor(rows: k, columns: n, rowBytes: n*2, dataType: .float16)
    let rdc = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n*2, dataType: .float16)
    let ba = dev.makeBuffer(length: m*k*2, options: .storageModeShared)!
    let bb = dev.makeBuffer(length: k*n*2, options: .storageModeShared)!
    let bc = dev.makeBuffer(length: m*n*2, options: .storageModeShared)!
    let A = MPSMatrix(buffer: ba, descriptor: rd)
    let B = MPSMatrix(buffer: bb, descriptor: rdb)
    let C = MPSMatrix(buffer: bc, descriptor: rdc)
    let mm = MPSMatrixMultiplication(device: dev, transposeLeft: false, transposeRight: false,
                                     resultRows: m, resultColumns: n, interiorColumns: k,
                                     alpha: 1.0, beta: 0.0)
    // warm
    for _ in 0..<3 { let cb = q.makeCommandBuffer()!; mm.encode(commandBuffer: cb, leftMatrix: A, rightMatrix: B, resultMatrix: C); cb.commit(); cb.waitUntilCompleted() }
    var best = Double.infinity
    for _ in 0..<5 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let cb = q.makeCommandBuffer()!
        for _ in 0..<iters { mm.encode(commandBuffer: cb, leftMatrix: A, rightMatrix: B, resultMatrix: C) }
        cb.commit(); cb.waitUntilCompleted()
        let dt = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters)
        best = min(best, dt)
    }
    return 2.0 * Double(m) * Double(k) * Double(n) / best / 1e12
}

print("--- fp16 matmul, TFLOP/s (MPS, best of 5) ---")
for (m,k,n,it) in [(4096,4096,4096,20),(2048,2048,2048,50),(8192,8192,8192,5),
                   (32,2560,2560,200),(32,2560,9728,200),(512,2560,9728,50),(128,2560,9728,100)] {
    print(String(format: "  M=%-5d K=%-5d N=%-5d  %8.2f TFLOP/s", m,k,n, gemm(m,k,n,iters:it)))
}

// bandwidth: large buffer copy via blit
print("--- memory bandwidth (blit copy, GB/s) ---")
let N = 1 << 29 // 512 MB
let src = dev.makeBuffer(length: N, options: .storageModePrivate)!
let dst = dev.makeBuffer(length: N, options: .storageModePrivate)!
var bestBW = 0.0
for _ in 0..<5 {
    let t0 = CFAbsoluteTimeGetCurrent()
    let cb = q.makeCommandBuffer()!
    let be = cb.makeBlitCommandEncoder()!
    for _ in 0..<4 { be.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: N) }
    be.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let dt = CFAbsoluteTimeGetCurrent() - t0
    bestBW = max(bestBW, Double(N)*4*2/dt/1e9)
}
print(String(format: "  read+write: %.1f GB/s", bestBW))
