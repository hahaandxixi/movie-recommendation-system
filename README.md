# 电影推荐系统

基于 MovieLens 1M 的课程设计项目，使用 Flask + MySQL 构建可交互的电影推荐网站，并融合热门推荐、ItemCF、UserCF、内容推荐、标签推荐和双塔模型。

## 功能

- 注册、登录、修改资料和交互历史
- 电影评分、收藏与浏览行为记录
- 个性化混合推荐与冷启动热门榜
- ItemCF 与内容相似电影解释
- 本地海报缓存及 TMDB 海报补全
- 离线 Precision@K、Recall@K、F1@K、Coverage、Novelty 评测

## 技术栈

- Python 3.10+
- Flask / Flask-WTF
- MySQL / PyMySQL
- NumPy / Pandas / PyTorch
- MovieLens 1M

## 快速开始

1. 克隆仓库并安装依赖：

   ```bash
   git clone https://github.com/hahaandxixi/movie-recommendation-system.git
   cd movie-recommendation-system
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. 创建本地配置：

   ```bash
   copy .env.example .env
   ```

   编辑 `.env`，至少填写 `MYIDEA_SECRET_KEY` 和 `MYIDEA_MYSQL_PASSWORD`。生成随机会话密钥：

   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

3. 启动 MySQL，然后运行：

   ```bash
   python app.py
   ```

4. 打开 `http://127.0.0.1:5000`。程序会自动创建数据库、表结构并导入电影信息。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MYIDEA_SECRET_KEY` | 随机临时值 | 正式运行应设置固定随机密钥 |
| `MYIDEA_MYSQL_HOST` | `127.0.0.1` | MySQL 地址 |
| `MYIDEA_MYSQL_PORT` | `3306` | MySQL 端口 |
| `MYIDEA_MYSQL_USER` | `root` | MySQL 用户 |
| `MYIDEA_MYSQL_PASSWORD` | 空 | MySQL 密码 |
| `MYIDEA_MYSQL_DATABASE` | `myidea_recommend` | 自动创建的数据库 |
| `MYIDEA_TMDB_KEY` | 空 | 可选的 TMDB API Key |
| `MYIDEA_FAST_START` | `0` | 设为 `1` 可跳过启动预热 |

## 项目结构

```text
core/       推荐算法、数据加载和双塔模型
models/     MySQL 数据访问和模型缓存
services/   用户、交互、推荐与海报服务
views/      Flask 蓝图
web/        应用工厂与安全配置
templates/  Jinja 页面模板
scripts/    海报构建与离线评测脚本
tests/      数据、算法和安全单元测试
```

## 测试与评测

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/run_benchmark.py
```

GitHub Actions 会在每个 Pull Request 上执行语法检查和单元测试。

## 安全说明

- `.env`、数据库和课程报告已被 Git 忽略。
- 新密码使用 Werkzeug 自适应带盐哈希保存。
- 旧版 SHA-256 密码会在用户成功登录时自动升级。
- 所有写操作表单均启用 CSRF 防护。

## 数据来源

项目使用 [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)。数据集仅用于课程学习和算法实验，使用时请遵守原始数据集许可与引用要求。
