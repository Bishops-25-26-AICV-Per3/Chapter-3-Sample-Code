import random

def print_problems(students: list[str]) -> list[(str, int, int, int, int)]:
    """Calculate output size given kernel, stride, channels.
    
    Calculates a different problem for each student.
    Return is correct answers."""
    answers = []
    print(f"{'Name':^20} Input  Kernel Stride Depth ")
    print("="*20, "="*6, "="*6, "="*6, "="*6)
    for student in students:
        in_size = random.randrange(180, 350)
        kernel = random.choice([5, 7, 9, 11, 13])
        stride = random.randrange(-1, 3) + kernel//3
        if stride < 2: stride = 2
        depth = random.choice([10, 12, 16, 24, 32])
        zero_pad = stride - ((in_size - kernel) % stride)
        if zero_pad == stride: zero_pad = 0
        out_size = (in_size - kernel + zero_pad)//stride + 1
        print(f"{student:>20}: {in_size:^6} {kernel:^6} {stride:^6} {depth:^6}")
        answers.append((student, zero_pad, out_size, out_size, depth))
    print()
    return answers

def print_answers(answers: list[(str, int, int, int, int)]) -> None:
    """Print formatted list of answers"""
    print(f"{'Name':^20} Zero-Pad {'Output Shape':^18}")
    print("="*20, "="*8, "="*18)
    for answer in answers:
        print(f"{answer[0]:>20}: {answer[1]:^8}  {answer[2]} x {answer[3]}", 
                f"x {answer[4]}")

def main():
    students = ["One", "Two", "Three"]
    answers = print_problems(students)
    input()
    print_answers(answers)

if __name__ == "__main__":
    main()