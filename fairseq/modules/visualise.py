from bertviz import model_view

def visualise(file, n):
    f = open(file, "r")
    for idx, line in f:
        if idx*2 == n:
            break
        model_view(attention, tokens)
    f.close()