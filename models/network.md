```mermaid
graph TD
A[输入图像] --> B[ClsModel.forward]
B --> C[VGG超列特征生成?]
C --> D[net_c(ConvNext)特征提取]
D --> E[FullNet_NLP.forward]
E --> F[stem_comp预处理]
F --> G[baseball(FocalNet)预训练特征]
G --> H[baseball_adapter通道适配]
H --> I[条件提示生成(prompt)]
I --> J[SubNet1处理]
J --> K[SubNet2处理]
K --> L[SubNetN处理]
L --> M[Decoder解码]
M --> N[输出clean/reflection]

subgraph ClsModel
    B --> C
    C --> D
    D --> E
end

subgraph FullNet_NLP
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
end

subgraph SubNet处理流程
    J --> P[Level0:上采样融合]
    P --> Q[ConvNextBlock特征增强]
    Q --> R[Level1:跨层连接]
    R --> S[Alpha加权残差]
    S --> T[递归处理]
end
```