# import sqlite3

# # 连接到 SQLite 数据库（如果数据库不存在，则会自动创建）
# conn = sqlite3.connect('papers.db')

# # 创建一个游标对象
# cursor = conn.cursor()

# # 创建 papers 表
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS papers (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     title TEXT NOT NULL,
#     abstract TEXT NOT NULL
# )
# ''')

# # 提交事务
# conn.commit()

# # 关闭连接
# conn.close()


# import sqlite3
# from papers import paper_base
# # 连接到 SQLite 数据库
# conn = sqlite3.connect('papers.db')

# # 创建一个游标对象
# cursor = conn.cursor()
# for paper in paper_base:
# # 插入数据
#     cursor.execute('''
#     INSERT INTO papers (title, abstract) VALUES (?, ?)
#     ''', (paper['name'], paper['abstract']))

# # 提交事务
# conn.commit()

# # 关闭连接
# conn.close()

import sqlite3

# 连接到 SQLite 数据库
conn = sqlite3.connect('papers.db')

# 创建一个游标对象
cursor = conn.cursor()

# 查询数据
cursor.execute('SELECT * FROM papers')

# 获取所有结果
rows = cursor.fetchall()

# 打印结果
for row in rows:
    print(f"ID: {row[0]}, Title: {row[1]}, Abstract: {row[2]}")

# 关闭连接
conn.close()