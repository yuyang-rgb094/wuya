# 两阶段路由架构：Router 与 CUDOS 职责分离

将 Main Agent 拆分为纯 Router（编排层）和独立 CUDOS Sub-agent，采用两阶段路由模式：阶段 1 仅调用 CUDOS 做门槛检查，通过后阶段 2 并行调用四个评价 Sub-agent。选择此方案是因为 Router 的调度职责与 CUDOS 的价值判断职责性质完全不同，合并会导致 prompt 互相干扰、难以独立迭代和调试；而拆分后 CUDOS 仍通过阶段 1 的否决权保留了"守门人"语义，投稿推荐场景还可跳过阶段 1 直接匹配。
