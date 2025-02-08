import openai
import base64
import os
import re
import json
import subprocess

from scrren_shotter import window_capture, get_all_digit_images, get_candidate_digit_position, get_sudoku_digits_positions, fill_sudoku
from mouse_controller import move_mouse, click_mouse, press_and_hold_mouse

bedrock_client = openai.OpenAI(api_key="anything", base_url="http://127.0.0.1:4000")
claude_3_5_v2_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
aliyun_api_key = os.environ.get("ALIYUN_API_KEY")
aliyun_client = openai.OpenAI(api_key=aliyun_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
aliyun_deepseek_model = "deepseek-r1"
qwen_vl_max_model = "qwen-vl-max"
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_client = openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
deepseek_r1_model = "deepseek-reasoner"

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

additional_prompts = []
while True:
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
    print(f"Original matrix: {matrix}")
    print("Start solving the sudoku")
    response = generate_solver_code(matrix, bedrock_client, claude_3_5_v2_model)
    print(f"Response: {response}")
    solved_matrix = json.loads(response)
    print(f"Solved matrix: {solved_matrix}")
    check_result = check_sudoku_correct(solved_matrix)
    print(f"Check result: {check_result}")
    if check_result == "Correct":
        break
    additional_prompts.append(f"Your answer: {json.dumps(matrix)} is error. {check_result}. Please try again.")
digit_pos_list = get_candidate_digit_position()
sukoku_digits_positions = get_sudoku_digits_positions()
fill_sudoku(matrix, solved_matrix, sukoku_digits_positions, digit_pos_list)