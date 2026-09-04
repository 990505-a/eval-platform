# eval-platform-guide

这是给 AI 使用的 `eval-platform` 操作 Skill。安装后，AI 可以根据当前平台的页面、运行方式和 Agent 接入契约，指导用户部署、配置、运行评测和排障。

## 下载内容

- `SKILL.md`：Skill 主文件，包含触发条件、操作流程、安全规则和故障处理。

## 安装

### Cursor 或支持项目 Skill 的客户端

将整个 `eval-platform-guide` 目录放入目标项目的 Skill 目录，例如：

```text
<项目根目录>/.cursor/skills/eval-platform-guide/SKILL.md
```

如果你的客户端使用其他 Skill 目录，以该客户端文档为准。安装后重新打开项目或新建一个 AI 会话，然后尝试：

```text
帮我用 eval-platform 跑一次 RAG 评测
帮我排查智能体没有接入的问题
帮我配置并运行 LLM 基准
```

### smart-test-platform

如果要上传到 `smart-test-platform` 的内置 Skill 管理器，请上传包含单一顶层目录的文件夹：

```text
eval-platform-guide/
└── SKILL.md
```

不要把包含 `.cursor/skills/` 的外层目录直接上传到该管理器；两种客户端的目录结构不同。

## 注意

- Skill 不包含 API Key、密码、用户数据或运行结果。
- 默认平台地址是 `http://127.0.0.1:8800`，实际地址以用户部署配置为准。
- 平台默认适合本机或可信网络，不要在没有认证和访问控制时直接暴露到公网。
- 真实模型、RAG 裁判、任务合成和红队扫描可能产生 API 费用。
