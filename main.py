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


def save_calculation(matrix, result, calculation_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{HISTORY_DIR}/calculation_{timestamp}.json"

    data = {
        'timestamp': timestamp,
        'type': calculation_type,
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
                # Форматируем данные для отображения в истории
                if data['type'] == 'matrix_game':
                    title = f"Матричная игра {data['timestamp']}"
                    result_summary = f"Цена игры: {data['result'].get('game_value', 'N/A')}"
                elif data['type'] == 'nature_game':
                    title = f"Игра с природой {data['timestamp']}"
                    criteria = [k for k in data['result'].keys() if k not in ['error']]
                    result_summary = f"Критерии: {', '.join(criteria)}"
                else:
                    title = f"Расчет {data['timestamp']}"
                    result_summary = "Результат расчета"

                history.append({
                    'id': filename.split('.')[0],
                    'title': title,
                    'date': datetime.strptime(data['timestamp'], "%Y%m%d_%H%M%S").strftime("%d.%m.%Y %H:%M"),
                    'result': result_summary,
                    'data': data
                })
    return history


def find_saddle_point(matrix):
    matrix_np = np.array(matrix)
    row_mins = np.min(matrix_np, axis=1)
    col_maxs = np.max(matrix_np, axis=0)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == row_mins[i] and matrix[i][j] == col_maxs[j]:
                return (i, j, matrix[i][j])
    return None


@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    try:
        matrix_rows = int(request.form.get('matrix_rows', 5))
        matrix_cols = int(request.form.get('matrix_cols', 5))
    except ValueError:
        matrix_rows = 5
        matrix_cols = 5

    matrix = [[0.0 for _ in range(matrix_cols)] for _ in range(matrix_rows)]
    result = None

    if request.method == 'POST' and 'cell_0_0' in request.form:
        try:
            # Считывание матрицы
            matrix = []
            for i in range(matrix_rows):
                row = []
                for j in range(matrix_cols):
                    value = request.form.get(f'cell_{i}_{j}', '0')
                    try:
                        row.append(float(value))
                    except ValueError:
                        row.append(0.0)
                matrix.append(row)

            matrix_np = np.array(matrix)

            # Поиск седловой точки
            saddle = find_saddle_point(matrix)

            # Приведение к положительным
            min_val = np.min(matrix_np)
            if min_val <= 0:
                matrix_np = matrix_np - min_val + 1

            # --- Игрок 1 (максимин) ---
            c = np.ones(matrix_rows)
            A_ub = -matrix_np  # (m x n)
            b_ub = -np.ones(matrix_cols)  # (n,)
            bounds = [(0, None)] * matrix_rows

            res = linprog(c, A_ub=A_ub.T, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                raise ValueError("Оптимальное решение не найдено для игрока 1")

            sum_probs = res.fun
            strategy1 = res.x / sum_probs
            game_value = 1 / sum_probs
            if min_val <= 0:
                game_value = game_value + min_val - 1

            # --- Игрок 2 (минимакс) ---
            c2 = -np.ones(matrix_cols)
            A_ub2 = matrix_np.T  # (n x m)
            b_ub2 = np.ones(matrix_rows) * game_value
            bounds2 = [(0, None)] * matrix_cols

            res2 = linprog(c2, A_ub=A_ub2.T, b_ub=b_ub2, bounds=bounds2, method='highs')

            if res2.success:
                sum_probs2 = -res2.fun
                strategy2 = res2.x / sum_probs2 if sum_probs2 > 0 else np.zeros(matrix_cols)
            else:
                strategy2 = np.zeros(matrix_cols)

            result = {
                "strategy1": [round(x, 4) for x in strategy1],
                "strategy2": [round(x, 4) for x in strategy2],
                "game_value": round(game_value, 4)
            }

            if saddle:
                i, j, value = saddle
                result['saddle_point'] = {
                    'i': i,
                    'j': j,
                    'value': round(value, 4)
                }

            save_calculation(matrix, result, 'matrix_game')

        except Exception as e:
            result = {
                'error': f"Ошибка при решении игры: {str(e)}"
            }

    return render_template('calculator.html',
                           matrix=matrix,
                           matrix_rows=matrix_rows,
                           matrix_cols=matrix_cols,
                           calculator_result=result)


@app.route('/nature_game', methods=['GET', 'POST'])
def nature_game():
    try:
        matrix_rows = int(request.form.get('matrix_rows', 4))
        matrix_cols = int(request.form.get('matrix_cols', 4))
        matrix = []

        for i in range(matrix_rows):
            row = []
            for j in range(matrix_cols):
                value = request.form.get(f'cell_{i}_{j}', '0')
                try:
                    row.append(float(value))
                except ValueError:
                    row.append(0.0)
            matrix.append(row)

        matrix_np = np.array(matrix)
        criteria = request.form.getlist('criteria')
        alpha = float(request.form.get('alpha', 0.5))
        result = {}

        if 'wald' in criteria:
            wald_values = np.min(matrix_np, axis=1)
            wald_best = np.argmax(wald_values)
            result['wald'] = {
                'values': [round(x, 4) for x in wald_values.tolist()],
                'best_strategy': int(wald_best) + 1,
                'best_value': round(float(wald_values[wald_best]), 4)
            }

        if 'savage' in criteria:
            regret_matrix = np.max(matrix_np, axis=0) - matrix_np
            savage_values = np.max(regret_matrix, axis=1)
            savage_best = np.argmin(savage_values)
            result['savage'] = {
                'values': [round(x, 4) for x in savage_values.tolist()],
                'best_strategy': int(savage_best) + 1,
                'best_value': round(float(savage_values[savage_best]), 4)
            }

        if 'hurwicz' in criteria:
            min_vals = np.min(matrix_np, axis=1)
            max_vals = np.max(matrix_np, axis=1)
            hurwicz_values = alpha * max_vals + (1 - alpha) * min_vals
            hurwicz_best = np.argmax(hurwicz_values)
            result['hurwicz'] = {
                'alpha': alpha,
                'values': [round(x, 4) for x in hurwicz_values.tolist()],
                'best_strategy': int(hurwicz_best) + 1,
                'best_value': round(float(hurwicz_values[hurwicz_best]), 4)
            }

        if 'laplace' in criteria:
            laplace_values = np.mean(matrix_np, axis=1)
            laplace_best = np.argmax(laplace_values)
            result['laplace'] = {
                'values': [round(x, 4) for x in laplace_values.tolist()],
                'best_strategy': int(laplace_best) + 1,
                'best_value': round(float(laplace_values[laplace_best]), 4)
            }

        if 'bayes' in criteria:
            # Предположим равновероятные вероятности состояний
            probabilities = np.ones(matrix_cols) / matrix_cols
            bayes_values = matrix_np.dot(probabilities)
            bayes_best = np.argmax(bayes_values)
            result['bayes'] = {
                'values': [round(x, 4) for x in bayes_values.tolist()],
                'best_strategy': int(bayes_best) + 1,
                'best_value': round(float(bayes_values[bayes_best]), 4),
                'probabilities': [round(x, 4) for x in probabilities.tolist()]
            }

        save_calculation(matrix, result, 'nature_game')

        return render_template('nature_game.html',
                               matrix=matrix,
                               matrix_rows=matrix_rows,
                               matrix_cols=matrix_cols,
                               result=result)
    except Exception as e:
        return render_template('nature_game.html',
                               matrix=[],
                               matrix_rows=0,
                               matrix_cols=0,
                               result={'error': f"Ошибка при расчёте: {str(e)}"})

@app.route('/history', methods=['GET', 'POST'])
def history():
    # Очистка истории, если был POST запрос с параметром 'clear_history'
    if request.method == 'POST' and 'clear_history' in request.form:
        for filename in os.listdir(HISTORY_DIR):
            file_path = os.path.join(HISTORY_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Ошибка при удалении файла {file_path}: {e}")
        return redirect(url_for('history'))

    # Загрузка и форматирование истории
    history_data = []
    for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if filename.endswith('.json'):
            try:
                with open(f"{HISTORY_DIR}/{filename}", 'r') as f:
                    data = json.load(f)

                    # Базовая информация
                    calculation = {
                        'id': filename.split('.')[0],
                        'timestamp': datetime.strptime(data['timestamp'], "%Y%m%d_%H%M%S").strftime("%d.%m.%Y %H:%M"),
                        'type': data.get('type', 'unknown'),
                        'matrix': data['matrix'],
                        'result': data['result']
                    }

                    # Детализация в зависимости от типа расчета
                    if calculation['type'] == 'matrix_game':
                        calculation['title'] = "Матричная игра"
                        calculation['summary'] = {
                            'game_value': data['result'].get('game_value', 'N/A'),
                            'has_saddle_point': 'saddle_point' in data['result']
                        }

                    elif calculation['type'] == 'nature_game':
                        calculation['title'] = "Игра с природой"
                        criteria = [k for k in data['result'].keys() if k not in ['error']]
                        calculation['summary'] = {
                            'criteria_used': criteria,
                            'best_strategies': {k: v['best_strategy'] for k, v in data['result'].items()
                                                if k not in ['error'] and 'best_strategy' in v}
                        }

                    else:
                        calculation['title'] = "Неизвестный расчет"
                        calculation['summary'] = {'result': 'Данные расчета'}

                    history_data.append(calculation)
            except Exception as e:
                print(f"Ошибка при загрузке файла {filename}: {str(e)}")

    # Отображение истории расчетов
    return render_template('history.html',
                           history=history_data,
                           now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


    return render_template('history.html',
                           history=history_data,
                           now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




@app.route('/')
def index():
    return redirect(url_for('calculator'))


if __name__ == '__main__':
    app.run(debug=True)