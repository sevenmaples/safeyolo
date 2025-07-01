import time
import functools

# 自动选择合适的时间单位，或强制指定单位
def _format_time(seconds, unit=None):
    if unit == "s":
        return f"{seconds:.6f} s"
    elif unit == "ms":
        return f"{seconds * 1e3:.3f} ms"
    elif unit == "us":
        return f"{seconds * 1e6:.1f} us"
    elif unit == "min":
        return f"{seconds / 60:.3f} min"
    elif unit == "h":
        return f"{seconds / 3600:.3f} h"
    # 自动模式
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.4f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} h"

def performance_measure(iterations=1, logger=None, unit=None):
    """
    性能测量装饰器。支持多次迭代、自动/自选单位、日志输出。
    :param iterations: 测量次数，默认1次
    :param logger: 日志记录器（可选），如不传则用print
    :param unit: 时间单位，可选'ns','us','ms','s','min','h'，默认自动
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None
            for i in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            avg = sum(times) / iterations
            unit_avg = _format_time(avg, unit)
            if iterations == 1:
                report = (
                    f"[性能测量报告] 函数: {func.__name__}\n"
                    f"  本次耗时: {unit_avg}\n"
                )
            else:
                report = (
                    f"[性能测量报告] 函数: {func.__name__}\n"
                    f"  迭代次数: {iterations}\n"
                    f"  平均耗时: {unit_avg}\n"
                )
            if logger:
                logger.info(report)
            else:
                print(report)
            return result
        return wrapper
    return decorator

# --- 用于测试 ---
if __name__ == '__main__':
    print("--- 单次执行 ---")
    @performance_measure(iterations=1)
    def test1():
        s = 0
        for i in range(100000):
            s += i
        return s
    test1()

    print("\n--- 多次执行 ---")
    @performance_measure(iterations=5)
    def test2():
        s = 0
        for i in range(100000):
            s += i
        return s
    test2()

    @performance_measure(iterations=5)  # 自动单位
    def foo(): ...

