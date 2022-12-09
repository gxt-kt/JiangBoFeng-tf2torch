验证方法:
1.利用之前在gitee上传的converth52txt.py转换了yolo_bench1.h5,命名为，yolo_tiny_bench1_afterprocess.bin作为参考标准；
2.然后使用load_weight2right_order.py，将yolo_bench1.h5转为pytorch可读形式yolo_tiny_weights_RIGHT.pth，
3.然后在yolo_input.py调用了现在用的pytorch库中的StoreWeights_BinConvert函数，对其进行转存为yolo_bench.bin。
4.利用diff进行比较，发现二者完全相同，即StoreWeights_BinConvert无误。
5.同理生成darknet19数据final_darknet_19_weights.bin,证明在proof.png



darknet19使用方法

1.readpre读取图片为8通道,多余通道为0。

2.load_weight2right_order为读取权重，改变字典名，生成符合顺序和名字的权重文件

3.darknet_19_model为读取输入，生成输出，包括第一层卷积、第26层卷积输出、最后一层detect层补充通道之后的输出，生成硬件所需权重文件。


