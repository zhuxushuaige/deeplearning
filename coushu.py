from typing import List, Tuple
from datetime import datetime

class SubsetSumFinder:
    def __init__(self, numbers: List[float], tolerance: float = 1e-6):
        """
        初始化求解器
        :param numbers: 输入的数字列表
        :param tolerance: 浮点数比较容差
        """
        self.numbers = numbers
        self.tolerance = tolerance
        self.results = []

    def find_combinations(self, target: float) -> List[Tuple[List[int], List[float], float]]:
        """
        查找所有和等于目标值的组合
        :param target: 目标和
        :return: 所有符合条件的组合，每项为 (索引列表, 数值列表, 实际和)
        """
        self.results = []
        self._backtrack(0, [], [], 0.0, target)
        return self.results

    def _backtrack(self, start: int, indices: List[int], values: List[float], 
                   current_sum: float, target: float):
        if abs(current_sum - target) < self.tolerance:
            self.results.append((indices.copy(), values.copy(), current_sum))
            return
        if current_sum > target + self.tolerance:
            return
        for i in range(start, len(self.numbers)):
            indices.append(i)
            values.append(self.numbers[i])
            self._backtrack(i + 1, indices, values, current_sum + self.numbers[i], target)
            indices.pop()
            values.pop()

    def print_results(self, show_all: bool = True, max_display: int = 100):
        """
        打印结果
        :param show_all: 是否显示所有结果
        :param max_display: 最多显示多少条结果
        """
        print(f"\n共找到 {len(self.results)} 种组合\n")
        if not self.results:
            print("未找到任何符合条件的组合。")
            return
        display_count = len(self.results) if show_all else min(max_display, len(self.results))
        for i, (indices, values, total) in enumerate(self.results[:display_count], 1):
            detail = " + ".join(f"{v:.2f}" for v in values)
            idx_detail = ", ".join(str(idx) for idx in indices)
            print(f"组合 {i}: [{detail}] = {total:.6f} (索引: {idx_detail})")
        if not show_all and len(self.results) > max_display:
            print(f"... 还有 {len(self.results) - max_display} 种组合未显示")

    def export_results(self, filename: str = "results.txt", target: float = None):
        """
        将结果导出到文件
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"凑数结果报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"目标值: {target}\n")
            f.write(f"输入数字: {self.numbers}\n")
            f.write(f"找到组合数: {len(self.results)}\n\n")
            for i, (indices, values, total) in enumerate(self.results, 1):
                f.write(f"组合 {i}:\n")
                f.write(f"  索引: {indices}\n")
                f.write(f"  数值: {values}\n")
                f.write(f"  总和: {total:.6f}\n\n")
        print(f"结果已保存到 {filename}")

# 示例用法
if __name__ == "__main__":
    # 示例数据（可替换为任意 25 个数字）
    numbers = [
        22208.03,-80026.6,60333.5,58178.74,43095.36,7541.69,34476.29,11851.22,9696.46,26934.6,99119.33,113738,115328,101760,130380,59503.1,552353.09,119609.52,21790.59,104657.19,10703.6,307106.06,6491.17,1097.94
    ]

    print("=== 凑数求解器 ===")
    print(f"共有 {len(numbers)} 个数字")
    
    try:
        target = float(input("请输入目标和值: "))
    except:
        target = 1446013.09
        print(f"使用默认目标值: {target}")

    print("\n正在搜索，请稍候...")
    start_time = datetime.now()

    finder = SubsetSumFinder(numbers)
    results = finder.find_combinations(target)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n搜索完成，耗时: {duration:.2f} 秒")
    finder.print_results(show_all=False, max_display=50)

    # 可选：导出结果到文件
    # finder.export_results("subset_sum_results.txt", target)