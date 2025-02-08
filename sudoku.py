import openai
import base64
import time
import re
import json
import copy
import subprocess

from scrren_shotter import window_capture, get_all_digit_images, get_candidate_digit_position, get_sudoku_digits_positions, fill_sudoku
from mouse_controller import move_mouse, click_mouse, press_and_hold_mouse

bedrock_client = openai.OpenAI(api_key="anything", base_url="http://127.0.0.1:4000")
claude_3_5_v2_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
aliyun_client = openai.OpenAI(api_key="sk-86242510cc3f431d986e30e5422db315", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
aliyun_deepseek_model = "deepseek-r1"
# aliyun_deepseek_model = "deepseek-r1-distill-llama-70b"
qwen_vl_max_model = "qwen-vl-max"
deepseek_client = openai.OpenAI(api_key="sk-37430ca7943a4d72b3acf9692d1beb4a", base_url="https://api.deepseek.com")
deepseek_r1_model = "deepseek-reasoner"
# model = "qwen-vl-ocr"
# model = "qwen-vl-max-latest"

def get_digit(encoded_image, client, model):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + encoded_image
                        },
                    },
                    {
                        "type": "text",
                        "text": """You are a helpful assistant that can extract digits from images. For this specific image only, please:
                        extract the digits from the image.
                        The digits are from 1 to 9.
                        If you can't see any digits, output 0.
"""
                    }

                ]
            }
        ],
    )
    text_response = response.choices[0].message.content
    return int(text_response)

def check_sudoku_correct(matrix):
    for row in matrix:
        if len(set(row)) != 9:
            return f"The row {row} is not correct"
    for col in range(9):
        if len(set([matrix[row][col] for row in range(9)])) != 9:
            return f"The col {[matrix[row][col] for row in range(9)]} is not correct"
    for i in range(3):
        for j in range(3):
            if len(set([matrix[row][col] for row in range(i*3, i*3+3) for col in range(j*3, j*3+3)])) != 9:
                return f"The subgrid {i*3+j} is not correct"
    return "Correct"

def solve_sudoku_with_llm(matrix, client, model, additional_prompts=[]):
    matrix_prompt = transform_sudoku_matrix_to_prompt(matrix)
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that can solve Sudoku puzzles. For this specific puzzle only, please:"
                            "solve the Sudoku puzzle."
                            "The Sudoku puzzle is a 9x9 grid with digits from 1 to 9."
                            "The Sudoku puzzle is guaranteed to have a unique solution."
                            + matrix_prompt +
                            "Please output the solution in the same format as the input."
                }

            ]
        }
    ]
    for additional_prompt in additional_prompts:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": additional_prompt
                    }
                ]
            }
        )
    print(f"Messages: {messages}")
    response = client.chat.completions.create(
        stream=True,
        model=model,
        temperature=0,
        messages=messages,
    )
    text_response = ""
    for chunk in response:
        print(f"{chunk.choices[0].delta.content}", end="")
        text_response += chunk.choices[0].delta.content
    return text_response

def generate_solver_code(matrix, client, model):
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful programmer that can generate code to solve Sudoku puzzles.:"
                    "Please generate python code to solve the Sudoku puzzle."
                    "The Sudoku puzzle is a 9x9 grid with digits from 1 to 9."
                    "The Sudoku puzzle is guaranteed to have a unique solution."
                    "The code should be runnable in terminal and has one argement which is the input matrix encoded as a json string."
                    "The code should print the solved matrix in the same format as the input."
                    "Output the code in <results> tag."
                }

            ]
        }
    ]
    for additional_prompt in additional_prompts:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": additional_prompt
                    }
                ]
            }
        )
    print(f"Messages: {messages}")
    response = client.chat.completions.create(
        stream=True,
        model=model,
        temperature=0,
        messages=messages,
    )
    text_response = ""
    for chunk in response:
        print(f"{chunk.choices[0].delta.content}", end="")
        if not chunk.choices[0].delta.content:
            break
        text_response += chunk.choices[0].delta.content
    code = re.search(r'<results>\n```python(.*)```\n</results>', text_response, re.DOTALL).group(1)
    with open("solver.py", "w") as f:
        f.write(code)
    solved_matrix = subprocess.run(["python", "solver.py", json.dumps(matrix)], capture_output=True).stdout.decode('utf-8')
    return solved_matrix

def transform_sudoku_response_to_matrix(response):
    matrix = []
    for row in response.split("\n"):
        if not row.startswith("|"):
            continue
        row_list = []
        for triple_digit in row.split("|")[1:-1]:
            for digit in triple_digit.split():
                if digit.strip() == ".":
                    print("Empty cell detected")
                    return None
                else:
                    row_list.append(int(digit.strip()))
        matrix.append(row_list)
    return matrix

def transform_sudoku_matrix_to_prompt(matrix):
    prompt = "+-------+-------+-------+\n"
    for row in range(9):
        prompt += "| "
        for col in range(9):
            if matrix[row][col] == 0:
                prompt += ". "
            else:
                prompt += f"{matrix[row][col]} "
            if col % 3 == 2:
                prompt += "| "
        prompt = prompt.strip()
        prompt += "\n"
        if row % 3 == 2:
            prompt += "+-------+-------+-------+\n"
    return prompt

def solve_sudoku(board):
    # 找到一个未填充的位置
    empty_cell = find_empty_cell(board)
    if not empty_cell:
        # 如果没有未填充的位置，说明数独已经解决
        return True
    row, col = empty_cell

    # 尝试填充 1 到 9 的数字
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            # 如果数字有效，则填充该数字
            board[row][col] = num

            # 递归解决剩余的数独
            if solve_sudoku(board):
                return True

            # 如果递归没有找到解决方案，则回溯
            board[row][col] = 0

    # 如果没有找到有效的数字，则返回 False
    return False

def find_empty_cell(board):
    # 遍历整个数独板，找到一个值为 0 的位置
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board, row, col, num):
    # 检查行是否有效
    for i in range(9):
        if board[row][i] == num:
            return False

    # 检查列是否有效
    for i in range(9):
        if board[i][col] == num:
            return False

    # 检查 3x3 子网格是否有效
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True



additional_prompts = []
while True:
    # image_path = f"/Users/tangjiee/Downloads/sudoku/{step_number}.png"
    # image_path = f"/Users/tangjiee/Downloads/sudoku/test.jpg"
    print("Start capturing the screen")
    image_path = f"screenshot.png"
    window_capture(window_name="iPhone Mirroring", image_path=image_path)
    print("Start getting sudoku digits")
    get_all_digit_images(image_path)
    matrix = []
    for row in range(9):
        row_list = []
        for col in range(9):
            digit_image_path = f"{image_path.split('.')[0]}_{row}_{col}.png"
            with open(digit_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            digit = get_digit(encoded_image, aliyun_client, qwen_vl_max_model)
            row_list.append(digit)
        matrix.append(row_list)
        print(f"Row {row}: {row_list}")
    # matrix = [[6, 0, 0, 0, 0, 8, 0, 0, 5], [0, 3, 8, 9, 0, 0, 0, 0, 0], [7, 0, 2, 0, 4, 1, 0, 9, 0], [0, 1, 7, 0, 0, 4, 3, 5, 0], [4, 2, 6, 7, 3, 0, 0, 0, 0], [0, 0, 5, 1, 0, 0, 4, 2, 0], [2, 0, 4, 8, 0, 6, 5, 0, 1], [0, 9, 0, 5, 0, 3, 2, 6, 0], [5, 0, 3, 0, 1, 0, 7, 0, 0]]
    # matrix = [[7, 9, 1, 0, 2, 5, 6, 0, 0], [2, 0, 0, 0, 0, 0, 0, 3, 9], [5, 3, 0, 9, 0, 8, 2, 0, 0], [0, 0, 0, 6, 0, 1, 0, 0, 0], [0, 2, 7, 8, 0, 0, 0, 0, 4], [8, 0, 0, 4, 0, 2, 0, 0, 7], [3, 0, 5, 0, 0, 0, 0, 0, 6], [0, 7, 0, 1, 3, 0, 0, 0, 5], [0, 8, 4, 0, 0, 0, 0, 0, 0]]
    print(f"Original matrix: {matrix}")
    print("Start solving the sudoku")
    response = generate_solver_code(matrix, bedrock_client, claude_3_5_v2_model)
    # response = solve_sudoku_with_llm(matrix, bedrock_client, claude_3_5_v2_model, additional_prompts=additional_prompts)
    # response = solve_sudoku_with_llm(matrix, deepseek_client, deepseek_r1_model, additional_prompts=additional_prompts)
    # response = solve_sudoku_with_llm(matrix, aliyun_client, aliyun_deepseek_model, additional_prompts=additional_prompts)
    print(f"Response: {response}")
    solved_matrix = json.loads(response)
    # solved_matrix = transform_sudoku_response_to_matrix(response)
    # solved_matrix = copy.deepcopy(matrix)
    # solve_sudoku(solved_matrix)
    print(f"Solved matrix: {solved_matrix}")
    check_result = check_sudoku_correct(solved_matrix)
    print(f"Check result: {check_result}")
    if check_result == "Correct":
        break
    additional_prompts.append(f"Your answer: {json.dumps(matrix)} is error. {check_result}. Please try again.")
digit_pos_list = get_candidate_digit_position()
sukoku_digits_positions = get_sudoku_digits_positions()
fill_sudoku(matrix, solved_matrix, sukoku_digits_positions, digit_pos_list)