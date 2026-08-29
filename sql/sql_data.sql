-- ============================================================
-- My_agent 企业签订进度智能查询系统数据库初始化脚本
-- 数据库：my_agent_db（企业签订进度 / 职员名单管理系统数据库）
-- 说明：支撑「意图识别 → 路由 → A2A Agent（Text-to-SQL → MCP → MySQL → 润色）→ 返回前端」
--       的查询 / 修改链路，共 2 张核心业务表：
--       employees     职员名单
--       sign_progress 签订进度表
-- 执行：mysql -uroot -p123456 < My_agent/sql/sql_data.sql
-- ============================================================

DROP DATABASE IF EXISTS my_agent_db;
CREATE DATABASE my_agent_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE my_agent_db;

-- ------------------------------------------------------------
-- 1. 职员名单表（查询职员信息时返回该职员所有相关信息）
-- ------------------------------------------------------------
CREATE TABLE employees (
    emp_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '职员ID，自增主键',
    emp_no VARCHAR(30) NOT NULL COMMENT '工号（如 E1001）',
    name VARCHAR(50) NOT NULL COMMENT '职员姓名',
    role VARCHAR(20) COMMENT '岗位（销售/技术/财务/法务/项目经理等）',
    dept VARCHAR(50) COMMENT '所属部门',
    phone VARCHAR(20) COMMENT '联系电话',
    email VARCHAR(100) COMMENT '邮箱',
    status VARCHAR(10) NOT NULL DEFAULT '在职' COMMENT '状态（在职/离职）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '入职时间',
    UNIQUE KEY unique_emp_no (emp_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='职员名单表';

-- ------------------------------------------------------------
-- 2. 签订进度表（查询签订进度 / 修改签订状态）
-- ------------------------------------------------------------
CREATE TABLE sign_progress (
    progress_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '进度ID，自增主键',
    company_name VARCHAR(100) NOT NULL COMMENT '客户公司名称（支持模糊查询）',
    project_name VARCHAR(200) NOT NULL COMMENT '项目名称',
    status VARCHAR(20) NOT NULL DEFAULT '洽谈中' COMMENT '签订状态（洽谈中/待签约/已签约/已驳回）',
    amount DECIMAL(14,2) COMMENT '合同金额（元）',
    person_in_charge VARCHAR(50) COMMENT '负责人（职员姓名）',
    start_date DATE COMMENT '启动日期',
    sign_date DATE COMMENT '签约日期',
    operator_name VARCHAR(50) COMMENT '最近操作人姓名',
    reason VARCHAR(200) COMMENT '最近操作原因（驳回必填）',
    remark TEXT COMMENT '备注',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    KEY idx_company (company_name),
    KEY idx_status (status),
    KEY idx_person (person_in_charge)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='签订进度表';

-- ------------------------------------------------------------
-- 3. 操作审计表（状态修改留痕）
-- ------------------------------------------------------------
CREATE TABLE audit_log (
    audit_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '审计ID，自增主键',
    progress_id INT NOT NULL COMMENT '进度ID（sign_progress.progress_id）',
    company_name VARCHAR(100) COMMENT '公司名称',
    old_status VARCHAR(20) COMMENT '原签订状态',
    new_status VARCHAR(20) COMMENT '新签订状态',
    operator_name VARCHAR(50) COMMENT '操作人姓名',
    reason VARCHAR(200) COMMENT '操作原因（驳回必填）',
    trace_id VARCHAR(32) COMMENT '请求追踪ID',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间',
    KEY idx_progress (progress_id),
    KEY idx_trace (trace_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='操作审计表';
