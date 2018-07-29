import TensorFlow
import Foundation

print("args: ", CommandLine.arguments)
print("env: ", ProcessInfo.processInfo.environment)
print("Hello TF Debug")

func emptyFunction(_ some: String) {
    print("nothin to do")
}

func opsStridedSliceTest(last value: Float) -> Tensor<Float> {
    let someA = value
    let someB: Float = 8.7
    let someC = 3.7 * someA + someB
    
    let image: [Float] = [01, 02, 03, 04, 05, 06, 07, 08, 09,
                          11, 12, 13, 14, 15, 16, 17, 18, 19,
                          21, 22, 23, 24, 25, 26, 27, 28, 29,
                          31, 32, 33, 34, 35, 36, 37, 38, someC]
    
    let tensor = Tensor<Float>(shape: TensorShape([4, 9]), scalars: image)
    let begin = Tensor<Int32>(shape: TensorShape([1]), scalars: [0])
    let strides = Tensor<Int32>(shape: TensorShape([1]), scalars: [3])
    let end = Tensor<Int32>(shape: TensorShape([1]), scalars: [Int32(image.count)])
    
    let result = Raw.stridedSlice(tensor,
                                  begin: begin,
                                  end: end,
                                  strides: strides,
                                  beginMask: 0,
                                  endMask: 0,
                                  ellipsisMask: 0,
                                  newAxisMask: 0,
                                  shrinkAxisMask: 0)
    
    let w = Tensor<Float>(shape: TensorShape([9, 2]), repeating: 0.01)
    let h = matmul(result, w)
    let z = sigmoid(h)
    
    return z
}

if CommandLine.arguments.count < 2 {
    print("Needs arg.")
    exit(0)
}

guard let arg = Float(CommandLine.arguments[1]) else { exit(0) }

let result = opsStridedSliceTest(last: arg)
print(result)
