## 1. todolist
    目前版本是single-class、single-label的，
    扩展多标签、多类别版本（激活函数、custom loss）

## multi-class & multi-label
    multi-class的mask是multi-channel的
    那么对应地，最后1*1 conv的channel维度应该是n_classes(single-class  --->  1)
    有两种情况：
    single-label：each pixel的各类别之间应该是竞争关系，那么activation应该用softmax，计算loss时可以加入交叉熵
    multi-label：有时物体之间有重叠，部分pixel拥有multi-label，那么activation应该用sigmoid，loss考虑二元交叉熵