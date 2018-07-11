# SwiftAI

Create your swift package programm:
```
mkdir SwiftAI
cd SwiftAI
swift package init --type executable
```

Set path for latest Swift for TensorFlow Toolchain.
```
export PATH=/Library/Developer/Toolchains/swift-latest/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin
```

Build your programm:
```
swift build -Xswiftc -O
```

You can review your graph in TensorBoard:
```
~/Library/Python/3.6/bin/tensorboard --logdir=~/Desktop/export
```
