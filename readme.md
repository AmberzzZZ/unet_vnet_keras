## model
    按照论文的示意图实现
    dropout: "Drop-out layers at the end of the contracting path perform further implicit data augmentation."
    pad: valid模式导致边缘损失，concatenate之前先crop特征图
    up-conv: upsampling(rise resolution) + conv(reduce channels)
    last layer: "At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes"

## todolist
    目前版本是single-class、single-label的，
    扩展多标签、多类别版本（激活函数、custom loss）

## multi-class & multi-label
    multi-class的mask是multi-channel的
    那么对应地，最后1*1 conv的channel维度应该是n_classes(single-class  --->  1)
    有两种情况：
    single-label：each pixel的各类别之间应该是竞争关系，那么activation可以用softmax，计算loss时可以加入交叉熵
    multi-label：有时物体之间有重叠，部分pixel拥有multi-label，那么activation应该用sigmoid，loss考虑二元交叉熵

## reweighting:
    reweighting bce: 正负样本unbalance
    bce_dice: 两个loss调整到相当的数量级
    reweighting dice: 可以对每个通道分别计算dice，然后加权求和