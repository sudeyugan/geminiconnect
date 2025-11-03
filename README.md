# API 服务使用指南

该项目是一个基于 Flask 的后端服务，提供对话接口并对接外部向量数据库进行检索与生成。包含网页交互页面（`templates/index.html`）、Web API（`app.py`）。

## 运行环境

- Windows
- Python 3.10/3.11（建议 3.11）
- 能访问向量库服务（默认：`http://10.1.0.220:9002/api`）

## 安装依赖（在项目根目录）

位置：`e:\junior_fisrtseme\SoftwareExplo\API\api`、

```bash
pip install -r requirements.txt
```

依赖：

- `flask==3.0.0`
- `flask-cors==4.0.0`
- `requests==2.31.0`

## 配置环境变量（可选）

可在当前命令行会话设置，未设置则使用 `config.py` 默认值。

```bash
set VECTOR_DB_BASE_URL=http://10.1.0.220:9002/api
```

```bash
set USER_NAME=Group4
```

```bash
set TOKEN=替换为你的有效令牌
```

说明：`set` 仅对当前会话有效，需要长期生效可使用 `setx`。

## 使用方式

### 确保依赖与环境变量已设置后，启动：

```bash
python app.py
```

### 访问网页

打开浏览器访问：

```bash
http://localhost:5000/
```
