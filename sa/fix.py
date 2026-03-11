import sys
sys.path.insert(0, ".")
import sensitivity_analysis as sa
import numpy as np

test_row = np.array([1.0, 1.0, -4.0, 0.5, 0.225, 0.225, 0.95])
result = sa._evaluate_base(test_row, 42)
print(result)