import re
from sympy import symbols, solve, simplify, fraction, Eq
from sympy.parsing.sympy_parser import parse_expr
from fractions import Fraction as Frac

# на js пока не переписал


def is_arithmetic(expression):
    return bool(re.search(r'[\d+\-*/()]', expression))


def is_fraction(expression):
    return bool(re.search(r'[\d+/()]', expression))


def is_equation(expression):
    exp = expression.replace(' ', '')
    if exp.count('=') != 1:
        return False
    left, right = exp.split('=')
    return bool(left and right and re.search('[a-zA-Z]', exp))


def is_quadratic(expression):
    return bool(re.search(r'\*\*2|\^2', expression))


def solve_arithmetic(expr):
    try:
        # заменяем ^ на ** для норм парсинга
        expr = expr.replace('^', '**')
        parsed = parse_expr(expr, evaluate=False)
        result = simplify(parsed)
        return str(float(result.evalf()))
    except:
        return "Не удалось вычислить выражение"


def solve_fraction(expr):
    try:

        if '/' in expr and not ('(' in expr or ')' in expr):
            parts = expr.split('/')
            if len(parts) == 2:
                simplified = Frac(int(parts[0]), int(parts[1]))
                return f"{simplified.numerator}/{simplified.denominator}"

        # обработка сложных дробных выражений
        expr = expr.replace('^', '**')
        parsed = parse_expr(expr, evaluate=False)
        simplified = simplify(parsed)
        num, den = fraction(simplified)
        return f"{num}/{den}" if den != 1 else str(num)
    except:
        return "Не удалось упростить дробь"


def solve_linear_equation(eq):
    # линейные уравнения
    try:
        eq = eq.replace('^', '**').replace('=', '==')
        x = symbols('x')
        parsed = parse_expr(eq, evaluate=False)
        solution = solve(parsed, x)
        return str(float(solution[0])) if solution else "Нет решения"
    except:
        return "Не удалось решить уравнение"


def solve_quadratic_equation(eq):
    # квадоатные
    try:
        eq = eq.replace('^', '**').replace('=', '==')
        x = symbols('x')
        parsed = parse_expr(eq, evaluate=False)
        solutions = solve(parsed, x)
        return [str(float(s.evalf())) for s in solutions] if solutions else "Нет решения"
    except:
        return "Не удалось решить уравнение"


def classify_task(task_obj):
    # определяем тип задачи

    task_text = task_obj.get('text', '').lower()
    task_examples = task_obj.get('examples', [])

    # по ключевым словам
    if any(word in task_text for word in ['уравнени', 'корни', 'корень']):
        if any(word in task_text for word in ['квадрат', 'корни']):
            return 'quadratic_equation'
        return 'linear_equation'

    if any(word in task_text for word in ['дробь', 'деление']):
        if 'сократ' in task_text:
            return 'simplified_fraction'
        return 'fraction'

    # по примерам с помощью регулярных выражений
    for example in task_examples:
        if is_equation(example):
            return 'quadratic_equation' if is_quadratic(example) else 'linear_equation'
        if is_fraction(example):
            return 'fraction'
        if is_arithmetic(example):
            return 'arithmetic'

    # если мы хз что это то кидаем чо хз
    return 'unknown'


def solve_task(task_obj):

    # основной метод

    task_type = classify_task(task_obj)
    solutions = []

    # по типу решаем задачу
    for example in task_obj.get('examples', []):
        try:
            if task_type == 'arithmetic':
                solutions.append(solve_arithmetic(example))
            elif task_type == 'fraction':
                solutions.append(solve_fraction(example))
            elif task_type == 'simplified_fraction':
                solutions.append(solve_fraction(example))
            elif task_type == 'linear_equation':
                solutions.append(solve_linear_equation(example))
            elif task_type == 'quadratic_equation':
                solutions.append(solve_quadratic_equation(example))
            else:
                solutions.append("Неизвестный тип задачи")
        except Exception as e:
            solutions.append(f"Ошибка при решении: {str(e)}")

    return {
        'type': task_type,
        'solutions': solutions,
        'original_examples': task_obj['examples']
    }


if __name__ == "__main__":
    # формат данных на вход - JSON!
    test_task = {
        'text': 'Решите квадратные уравнения',
        'examples': [
            'x^2 + 2x + 1 = 0',
            'x^2 - 4x + 4 = 0',
            '2*x + 4 = 10'
        ]
    }
    print(solve_task(test_task))
