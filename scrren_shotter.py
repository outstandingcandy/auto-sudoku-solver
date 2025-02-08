import pyautogui
import pygetwindow as gw
from PIL import ImageGrab

import cv2 as cv
import numpy as np
import time
from mss import mss
from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
import Quartz
from mouse_controller import move_mouse, click_mouse, press_and_hold_mouse

def get_window_dimensions(hwnd):
    window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionIncludingWindow, hwnd)
    for window_info in window_info_list:
        window_id = window_info[Quartz.kCGWindowNumber]
        if window_id == hwnd:
            bounds = window_info[Quartz.kCGWindowBounds]
            width = bounds['Width']
            height = bounds['Height']
            left = bounds['X']
            top = bounds['Y']
            return {"top": top, "left": left, "width": width, "height": height}
    return None


def window_capture(window_name, image_path="screenshot.png"):
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    for window in window_list:
        # print(window.get('kCGWindowName', ''))
        if window_name.lower() == window.get('kCGWindowName', '').lower():
            hwnd = window['kCGWindowNumber']
            print('found window id %s' % hwnd)
    monitor = get_window_dimensions(hwnd)
    with mss() as sct:
        screenshot = np.array(sct.grab(monitor))
        # screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
        # cv.imshow('Computer Vision', screenshot)
        # resize the image
        # screenshot = cv.resize(screenshot, (342//2, 758//2))
        # save the image
        cv.imwrite(image_path, screenshot)

def clip_image(input_image_path, output_image_path, x, y, width, height):
    image = cv.imread(input_image_path)
    clip = image[y:y+height, x:x+width]
    cv.imwrite(output_image_path, clip)

def get_all_digit_images(image_path):
    x_offset = 25
    y_offset = 390
    for row in range(9):
        for col in range(9):
            x = 70 * col
            y = 70 * row
            clip_image(image_path, f"{image_path.split(".")[0]}_{row}_{col}.png", x_offset+x, y_offset+y, 70, 70)

def get_sudoku_digits_positions():
    digit_x = 25
    digit_y = 235
    digit_pos_max = []
    for row in range(9):
        digit_pos_list = []
        for col in range(9):
            digit_pos_list.append((digit_x + 35 * col, digit_y + 35 * row))
        digit_pos_max.append(digit_pos_list)
    return digit_pos_max

def get_candidate_digit_position():
    candidate_digit_x = 30
    candidate_digit_y = 670
    candidate_digit_pos_list = []
    for i in range(9):
        candidate_digit_pos_list.append((candidate_digit_x + 35 * i, candidate_digit_y))
    return candidate_digit_pos_list

def fill_sudoku(original_matrix, solved_matrix, sudoku_digits_positions, candidate_digit_pos_list):
    move_mouse(100, 100)
    click_mouse()
    for i in range(9):
        for j in range(9):
            if original_matrix[i][j] == 0:
                print(f"Fill in the digit at row {i} and column {j}")
                print(f"Move mouse to position {sudoku_digits_positions[i][j]}")
                move_mouse(sudoku_digits_positions[i][j][0], sudoku_digits_positions[i][j][1])
                click_mouse()
                time.sleep(1)
                print(f"Select the digit {solved_matrix[i][j]}")
                print(f"Move mouse to position {candidate_digit_pos_list[solved_matrix[i][j] - 1]}")
                move_mouse(candidate_digit_pos_list[solved_matrix[i][j] - 1][0], candidate_digit_pos_list[solved_matrix[i][j] - 1][1])
                click_mouse()
                time.sleep(1)

if __name__ == "__main__":
    window_name = "iPhone Mirroring"
    # window_capture(window_name=window_name)
    # clip_image("screenshot.png", 0, 384, 684, 78)
    # get_all_digits("screenshot.png")
    digit_pos_list = get_candidate_digit_position()
    sukoku_digits_positions = get_sudoku_digits_positions()
    original_matrix = [[6, 0, 0, 0, 0, 8, 0, 0, 5], [0, 3, 8, 9, 0, 0, 0, 0, 0], [7, 0, 2, 0, 4, 1, 0, 9, 0], [0, 1, 7, 0, 0, 4, 3, 5, 0], [4, 2, 6, 7, 3, 0, 0, 0, 0], [0, 0, 5, 1, 0, 0, 4, 2, 0], [2, 0, 4, 8, 0, 6, 5, 0, 1], [0, 9, 0, 5, 0, 3, 2, 6, 0], [5, 0, 3, 0, 1, 0, 7, 0, 0]]
    solved_matrix = [[6, 4, 9, 3, 2, 8, 1, 7, 5], [1, 3, 8, 9, 5, 7, 6, 4, 2], [7, 5, 2, 6, 4, 1, 8, 9, 3], [9, 1, 7, 2, 8, 4, 3, 5, 6], [4, 2, 6, 7, 3, 5, 9, 1, 8], [3, 8, 5, 1, 6, 9, 4, 2, 7], [2, 7, 4, 8, 9, 6, 5, 3, 1], [8, 9, 1, 5, 7, 3, 2, 6, 4], [5, 6, 3, 4, 1, 2, 7, 8, 9]]
    fill_sudoku(original_matrix, solved_matrix, sukoku_digits_positions, digit_pos_list)

