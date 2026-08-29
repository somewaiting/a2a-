#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: my_agent_service.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 企业签订进度数据服务，封装 MySQL 连接池、查询与状态修改。
      安全：查询强行限定仅 SELECT + 强白名单（禁多语句/注释/UNION/非白名单表/非法列），禁止任何写操作。
      一致性：状态修改走专用接口并带乐观锁（并发下不覆盖他人修改）。
      审计：状态修改成功落库 audit_log（前后状态/操作人/原因/trace_id）。
"""

import json
import re
from mysql.connector import pooling
from datetime import date, datetime, timedelta
from decimal import Decimal

from My_agent.config import Config
from My_agent.create_logger import logger
from My_agent.utils.format import DateEncoder, default_encoder

conf = Config()

# 签订状态枚举
STATUS_LIST = ['洽谈中', '待签约', '已签约', '已驳回']

# 合法表名（白名单）
ALLOWED_TABLES = {'sign_progress', 'employees', 'audit_log'}

# 合法列名（白名单）
ALLOWED_COLUMNS = {
    # sign_progress
    'progress_id', 'company_name', 'project_name', 'status', 'amount',
    'person_in_charge', 'start_date', 'sign_date', 'operator_name', 'reason', 'remark', 'updated_at',
    # employees
    'emp_id', 'emp_no', 'name', 'role', 'dept', 'phone', 'email',
    # audit_log（审计查询）
    'audit_id', 'old_status', 'new_status', 'trace_id', 'created_at',
}

# 危险关键字/函数（黑名单兜底）
FORBIDDEN_KEYWORDS = [
    'DELETE', 'UPDATE', 'INSERT', 'DROP', 'TRUNCATE', 'ALTER', 'GRANT',
    'REVOKE', 'RENAME', 'CREATE ', 'SET ',
    'UNION', 'INTO OUTFILE', 'LOAD_FILE', 'SLEEP', 'BENCHMARK',
    'INFORMATION_SCHEMA', 'mysql.', 'INTO ',
]

# 注释标记
COMMENT_MARKERS = ['--', '#', '/*', '*/']


class MyAgentService:
    def __init__(self):
        # 连接池：多请求并发时各取各的连接，避免共享单连接
        self.pool = pooling.MySQLConnectionPool(
            pool_name="my_agent_pool",
            pool_size=5,
            host=conf.host,
            port=conf.port,
            user=conf.user,
            password=conf.password,
            database=conf.database,
            pool_reset_session=True,
        )

    def _get_conn(self):
        return self.pool.get_connection()

    # ========== SQL 强白名单校验 ==========
    def _validate_select_sql(self, sql: str) -> str | None:
        """校验 SELECT 是否安全。返回 None 表示通过，否则返回拒绝消息。"""
        stripped = sql.strip().rstrip(';').strip()
        # 1) 仅允许单条 SELECT
        if not re.match(r'^SELECT\b', stripped, re.IGNORECASE):
            return "请联系相关负责人处理"
        # 2) 禁多语句（去除末尾分号后不得再出现分号）
        if ';' in stripped:
            return "请联系相关负责人处理"
        # 3) 禁注释
        upper = stripped.upper()
        for marker in COMMENT_MARKERS:
            if marker in upper:
                return "请联系相关负责人处理"
        # 4) 禁危险关键字/函数
        for kw in FORBIDDEN_KEYWORDS:
            if kw in upper:
                return "请联系相关负责人处理"
        # 5) 表白名单：FROM/JOIN 涉及的表必须合法
        tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-zA-Z_]+)', stripped, re.IGNORECASE))
        for t in tables:
            if t.lower() not in ALLOWED_TABLES:
                return "请联系相关负责人处理"
        # 6) 列白名单：SELECT 列必须在合法列集合内
        m = re.match(r'^SELECT\s+(.+?)\s+FROM\b', stripped, re.IGNORECASE)
        if m:
            for col in self._extract_columns(m.group(1)):
                if col.lower() not in ALLOWED_COLUMNS:
                    return "请联系相关负责人处理"
        return None

    @staticmethod
    def _extract_columns(select_part: str):
        """提取 SELECT 子句中的列名（去除表别名/AS 别名/聚合函数包裹）。"""
        cols = []
        for part in select_part.split(','):
            p = part.strip()
            p = re.sub(r'\s+AS\s+[`\w.]+', '', p, flags=re.IGNORECASE).strip()  # 去 AS 别名
            fm = re.match(r'^[A-Za-z_]+\((.+)\)$', p, re.IGNORECASE)  # 聚合函数 COUNT(x)
            if fm:
                p = fm.group(1).strip()
            if '.' in p:  # 去表别名前缀 e.emp_id
                p = p.split('.')[-1].strip()
            if p == '*' or p == '':
                continue
            cols.append(p)
        return cols

    # ========== 只读查询 ==========
    def _exec_select(self, sql: str) -> str:
        """实际执行 SELECT 并返回 JSON 字符串。"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)
            return json.dumps({"status": "success", "data": results} if results else {"status": "no_data",
                                                                                      "message": "无相关信息，请联系相关负责人处理或查找其他内容"},
                              cls=DateEncoder, ensure_ascii=False)
        except Exception as e:
            logger.error(f"查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": f"查询出错：{str(e)}"}, ensure_ascii=False)
        finally:
            conn.close()

    def _fuzzy_fallback(self, sql: str) -> str | None:
        """模糊查询兜底：当 LIKE '%关键词%' 无结果时，逐步缩短关键词重试（如「西湖文旅」→「西湖」）。"""
        m = re.search(r"LIKE\s+'%([^']+)%'", sql, re.IGNORECASE)
        if not m:
            return None
        kw = m.group(1)
        for new_len in range(len(kw) - 1, 1, -1):
            candidate = kw[:new_len]
            new_sql = re.sub(r"LIKE\s+'%[^']*%'", f"LIKE '%{candidate}%'", sql, count=1, flags=re.IGNORECASE)
            r = self._exec_select(new_sql)
            if json.loads(r).get("status") == "success":
                logger.info(f"模糊兜底命中: {kw} -> {candidate}")
                return r
        return None

    def execute_query(self, sql: str) -> str:
        """执行 SELECT 查询，返回 JSON 字符串 {status, data/message}。
        强白名单：仅允许单条 SELECT、禁注释/危险关键字、仅白名单表与列。
        支持模糊兜底：无结果时自动缩短 LIKE 关键词重试。"""
        try:
            reject_msg = self._validate_select_sql(sql)
            if reject_msg:
                logger.warning(f"SQL 校验被拒绝: {sql}")
                return json.dumps({"status": "rejected", "message": reject_msg}, ensure_ascii=False)

            stripped = sql.strip().rstrip(';').strip()
            result = self._exec_select(stripped)
            parsed = json.loads(result)
            if parsed.get("status") == "no_data":
                fallback = self._fuzzy_fallback(stripped)
                if fallback is not None:
                    return fallback
            return result
        except Exception as e:
            logger.error(f"查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": f"查询出错：{str(e)}"}, ensure_ascii=False)

    # ========== 状态修改（专用接口 + 乐观锁 + 审计） ==========
    def update_sign_status(self, target, new_status: str, operator_name: str, reason: str = '',
                           trace_id: str = '', expected_old_status: str = None) -> str:
        """修改签订进度状态。target 为公司名或进度ID；驳回（已驳回）必须提供操作人姓名与原因。
        expected_old_status 供乐观锁测试/调用方显式校验：若不匹配当前状态则拒绝修改。"""
        # 参数校验
        if new_status not in STATUS_LIST:
            return json.dumps({"status": "error",
                               "message": f"状态【{new_status}】非法，仅支持：{STATUS_LIST}"}, ensure_ascii=False)
        if not operator_name:
            return json.dumps({"status": "error", "message": "修改签订状态需提供操作人姓名。"}, ensure_ascii=False)
        if new_status == '已驳回' and not reason:
            return json.dumps({"status": "error", "message": "驳回操作必须提供原因，请说明驳回原因。"}, ensure_ascii=False)

        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            # 定位进度记录（支持公司名模糊 或 进度ID）
            if str(target).isdigit():
                cursor.execute("SELECT * FROM sign_progress WHERE progress_id=%s", (int(target),))
            else:
                cursor.execute("SELECT * FROM sign_progress WHERE company_name LIKE %s LIMIT 1", (f"%{target}%",))
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return json.dumps({"status": "no_data", "message": "无相关信息，请联系相关负责人处理或查找其他内容"}, ensure_ascii=False)
            progress_id = row["progress_id"]
            old_status = row["status"]

            # 乐观锁：若调用方显式给定期望旧状态，先校验；否则以读到的旧状态作为 WHERE 条件
            where_status = expected_old_status if expected_old_status is not None else old_status
            if expected_old_status is not None and old_status != expected_old_status:
                cursor.close()
                return json.dumps({"status": "error",
                                   "message": f"状态已被他人修改（当前为【{old_status}】，期望【{expected_old_status}】），请重新查询后再操作。"},
                                  ensure_ascii=False)

            # 执行条件更新（防并发覆盖）
            if new_status == '已签约':
                sql = ("UPDATE sign_progress SET status=%s, operator_name=%s, reason=%s, sign_date=CURDATE() "
                       "WHERE progress_id=%s AND status=%s")
            else:
                sql = "UPDATE sign_progress SET status=%s, operator_name=%s, reason=%s WHERE progress_id=%s AND status=%s"
            cursor.execute(sql, (new_status, operator_name, reason or None, progress_id, where_status))

            if cursor.rowcount == 0:
                conn.rollback()
                cursor.close()
                return json.dumps({"status": "error",
                                   "message": "状态已被他人修改，请重新查询后再操作。"}, ensure_ascii=False)

            conn.commit()

            # 审计落库
            audit_sql = ("INSERT INTO audit_log (progress_id, company_name, old_status, new_status, "
                         "operator_name, reason, trace_id) VALUES (%s, %s, %s, %s, %s, %s, %s)")
            cursor.execute(audit_sql, (progress_id, row["company_name"], old_status, new_status,
                                       operator_name, reason or None, trace_id or None))
            conn.commit()
            cursor.close()

            return json.dumps({"status": "success",
                               "message": f"已将【{row['company_name']}】的签订状态修改为【{new_status}】"
                                          + (f"，操作人：{operator_name}，原因：{reason}。" if reason else f"，操作人：{operator_name}。")},
                              ensure_ascii=False)
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"状态修改错误: {str(e)}")
            return json.dumps({"status": "error", "message": f"状态修改失败：{str(e)}"}, ensure_ascii=False)
        finally:
            conn.close()


if __name__ == '__main__':
    svc = MyAgentService()
    print(svc.execute_query("SELECT company_name, project_name, status FROM sign_progress"))
