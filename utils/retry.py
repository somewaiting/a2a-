#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: retry.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 通用指数退避重试工具（异步）
"""
import asyncio


async def with_retry(fn, retries: int = 2, base_delay: float = 0.5, on_retry=None):
    """
    异步执行 fn() 并做指数退避重试。
    :param fn: 无参异步可调用对象（已绑定参数的 coroutine function 或 lambda）
    :param retries: 失败后最多重试次数（总尝试 = retries + 1）
    :param base_delay: 首次重试等待秒数，之后每次翻倍
    :param on_retry: 可选回调 on_retry(attempt_index, exception)
    :return: fn() 的返回值
    :raises: 最后一次异常（重试耗尽后）
    """
    delay = base_delay
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            if attempt >= retries:
                break
            if on_retry:
                on_retry(attempt, e)
            await asyncio.sleep(delay)
            delay *= 2
    raise last_exc
