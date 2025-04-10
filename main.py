from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import linprog

app = Flask(__name__)

game_description = """
Каждый игрок выбирает уровень сотрудничества от 1 (предательство) до 5 (полное доверие). Выплаты зависят от комбинации: чем выше уровень доверия, тем выше риск, но и выше выигрыш.
"""

def find_saddle_point(matrix):
    matrix_np = np.array(matrix)
    row_mins = np.min(matrix_np, axis=1)
    col_maxs = np.max(matrix_np, axis=0)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == row_mins[i] and matrix[i][j] == col_maxs[j]:
                return (i, j, matrix[i][j])  # возвращаем координаты и значение
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    matrix_size = 5
    matrix = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    result = None

    if request.method == 'POST':
        try:
            matrix_size = int(request.form.get('matrix_size', 5))
            matrix = []
            for i in range(matrix_size):
                row = []
                for j in range(matrix_size):
                    value = request.form.get(f'cell_{i}_{j}', '0')
                    try:
                        row.append(float(value))
                    except ValueError:
                        row.append(0.0)
                matrix.append(row)

            matrix_np = np.array(matrix)

            # Проверка на наличие седловой точки
            saddle = find_saddle_point(matrix)

            # Приводим матрицу к положительным значениям
            min_val = np.min(matrix_np)
            if min_val <= 0:
                matrix_np = matrix_np - min_val + 1

            c = np.ones(matrix_size)
            A_ub = -matrix_np.T
            b_ub = -np.ones(matrix_size)
            bounds = [(0, None)] * matrix_size

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                raise ValueError("Оптимальное решение не найдено для игрока 1")

            sum_probs = res.fun
            strategy1 = res.x / sum_probs
            game_value = 1 / sum_probs
            if min_val <= 0:
                game_value = game_value + min_val - 1

            c2 = -np.ones(matrix_size)
            A_ub2 = matrix_np
            b_ub2 = np.ones(matrix_size) * game_value
            bounds2 = [(0, None)] * matrix_size

            res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')

            if res2.success:
                sum_probs2 = -res2.fun
                strategy2 = res2.x / sum_probs2 if sum_probs2 > 0 else np.zeros(matrix_size)
            else:
                strategy2 = np.zeros(matrix_size)

            result = {
                "strategy1": [round(x, 2) for x in strategy1],
                "strategy2": [round(x, 2) for x in strategy2],
                "game_value": round(game_value, 2)
            }

            if saddle:
                i, j, value = saddle
                result['saddle_point'] = {
                    'i': i,
                    'j': j,
                    'value': value
                }

        except Exception as e:
            result = {
                'error': f"Ошибка при решении игры: {str(e)}"
            }

    # Возвращаем данные в шаблон
    return render_template('index.html',
                           matrix=matrix,  # передаем обычную матрицу, а не NumPy
                           matrix_size=matrix_size,
                           description=game_description,
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)
