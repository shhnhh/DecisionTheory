from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from scipy.optimize import linprog
import os
import json
from datetime import datetime

app = Flask(__name__)

# Конфигурация
HISTORY_DIR = 'history'
os.makedirs(HISTORY_DIR, exist_ok=True)


def save_calculation(matrix, result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{HISTORY_DIR}/calculation_{timestamp}.json"

    data = {
        'timestamp': timestamp,
        'matrix': matrix,
        'result': result
    }

    with open(filename, 'w') as f:
        json.dump(data, f)

    return filename


def load_history():
    history = []
    for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if filename.endswith('.json'):
            with open(f"{HISTORY_DIR}/{filename}", 'r') as f:
                data = json.load(f)
                history.append(data)
    return history


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
    history = load_history()

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

            # Сохраняем расчет в историю
            save_calculation(matrix, result)
            history = load_history()  # Обновляем историю

        except Exception as e:
            result = {
                'error': f"Ошибка при решении игры: {str(e)}"
            }

    return render_template('index.html',
                           matrix=matrix,
                           matrix_size=matrix_size,
                           result=result,
                           history=history)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    for filename in os.listdir(HISTORY_DIR):
        file_path = os.path.join(HISTORY_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)