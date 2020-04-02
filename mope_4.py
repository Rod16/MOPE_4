import math
from scipy.stats import f, t
from functools import partial
from random import randint
import numpy as np
from prettytable import PrettyTable
from numpy.linalg import solve

def cochrane(g_prac, g_teor):
    return g_prac < g_teor


def student(t_teor, t_pr):
    return t_pr < t_teor


def fisher(f_teor, f_prac):
    return f_teor > f_prac


def cochrane_teor(f1, f2, q=0.05):
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)

def cohrane_t(f1, f2, q = 0.05):
	q1 = q / f1

fisher_t = partial(f.ppf, q=1 - 0.05)
student_t = partial(t.ppf, q=1 - 0.025)

n = 8
m = 3

x1min = -25
x1max = 75
x2min = 25
x2max = 65
x3min = 25
x3max = 40

xmin = (x1min + x2min + x3min)/3
xmax = (x1max + x2max + x3max)/3

ymin = round(200 + xmin)
ymax = round(200 + xmax)

while True:
	x0f = [1, 1, 1, 1, 1, 1, 1, 1]
	x1f = [-1, -1, -1, -1, 1, 1, 1, 1]
	x2f = [-1, -1, 1, 1, -1, -1, 1, 1]
	x3f = [-1, 1, -1, 1, -1, 1, -1, 1]
	x12f = [i*j for i, j in zip(x1f, x2f)]
	x13f = [i*j for i, j in zip(x1f, x3f)]
	x23f = [i*j for i, j in zip(x2f, x3f)]
	x123f = [i*j*k for i, j, k in zip(x1f, x2f, x3f)]

	list_fact = [x0f, x1f, x2f, x3f, x12f, x13f, x23f, x123f]

	y1 = [randint(ymin, ymax) for i in range(n)]
	y2 = [randint(ymin, ymax) for i in range(n)]
	y3 = [randint(ymin, ymax) for i in range(n)]

	y_rows = {}
	for i in range(1, 9):
		y_rows["Рядок{0}".format(i)] = [y1[i-1], y2[i-1], y3[i-1]]

	y_row_ser = {}
	for i in range(1, 9):
		y_row_ser["Середнє значення Y у рядку{0}".format(i)] = np.average(y_rows[f"Рядок{i}"])
	y_ser = [round(val, 3) for val in y_row_ser.values()]

	x0 = [1, 1, 1, 1, 1, 1, 1, 1]
	x1 = [x1min, x1min, x1max, x1max, x1min, x1min, x1max, x1max]
	x2 = [x2min, x2max, x2min, x2max, x2min, x2max, x2min, x2max]
	x3 = [x3min, x3max, x3max, x3min, x3max, x3min, x3min, x3max]
	x1x2 = [a * b for a, b in zip(x1, x2)]
	x1x3 = [a * b for a, b in zip(x1, x3)]
	x2x3 = [a * b for a, b in zip(x2, x3)]
	x1x2x3 = [a * b * c for a, b, c in zip(x1, x2, x3)]
	x_arr = [x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3]

	list_x = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))

	list_bi = []
	for k in range(n):
		S = 0
		for i in range(n):
		    S += (list_fact[k][i] * y_ser[i]) / n
		list_bi.append(round(S, 5))

	disp = {}
	for i in range(1, 9):
		disp["Дисперсія{0}".format(i)] = 0
	for i in range(m):
		ctr = 1
		for key, value in disp.items():
			row = y_rows[f'Рядок{ctr}']
			disp[key] += ((row[i] - np.average(row)) ** 2) / m
			ctr += 1
	disp_sum = sum(disp.values())
	disp_list = [round(disp, 3) for disp in disp.values()]

	column = ["x0", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "y1", "y2", "y3", "y", "s^2"]

	pt = PrettyTable()
	list_fact.extend([y1, y2, y3, y_ser, disp_list])
	for k in range(len(list_fact)):
	    pt.add_column(column[k], list_fact[k])

	print(pt, "\n")
	# Regression eq with interaction effect
	print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(list_bi[0], list_bi[1],
	                                                                                           list_bi[2], list_bi[3],
	                                                                                           list_bi[4], list_bi[5],
	                                                                                           list_bi[6], list_bi[7]))

	pt = PrettyTable()
	x_arr.extend([y1, y2, y3, y_ser, disp_list])
	for k in range(len(list_fact)):
	    pt.add_column(column[k], list_fact[k])
	print(pt, "\n")

	list_ai = [round(i, 5) for i in solve(list_x, y_ser)]
	print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(list_ai[0], list_ai[1],
	                                                                                        list_ai[2], list_ai[3],
	                                                                                        list_ai[4], list_ai[5],
	                                                                                        list_ai[6], list_ai[7]))

	gp = max(disp.values()) / disp_sum
	f1 = m -1
	f2 = n
	gt = cochrane_teor(f1, f2)

	if cochrane(gp, gt):
	    print("Дисперсія є однорідною\n")

	    dispersion_b = disp_sum / n
	    dispersion_beta = dispersion_b / (m * n)
	    s_beta = math.sqrt(abs(dispersion_beta))

	    beta = {}
	    for x in range(8):
	        beta["beta{0}".format(x)] = 0
	    for i in range(len(x0f)):
	        ctr = 0
	        for key, value in beta.items():
	            beta[key] += (y_ser[i] * list_fact[ctr][i]) / n
	            ctr += 1

	    beta_list = list(beta.values())
	    t_list = [abs(k) / s_beta for k in beta_list]

	    f3 = f1 * f2
	    d = 0
	    t = student_t(df=f3)
	    print("t = ", t)
	    for i in range(len(t_list)):
	        if student(t_list[i], t):
	            beta_list[i] = 0
	            print("Коефцієнт не є значущим, beta{} = 0".format(i))
	        else:
	            print("Коефіцієнт є значущим, beta{} = {}".format(i, beta_list[i]))
	            d += 1

	    list_fact[0] = None
	    y_student = [sum([a * b[x_idx] if b else a for a, b in zip(beta_list, list_x)]) for x_idx in range(8)]

	    f4 = n - d
	    dispersion_ad = 0
	    for i in range(len(y_student)):
	        dispersion_ad += ((y_student[i] - y_ser[i]) ** 2) * m / (n - d)
	    fp = dispersion_ad / dispersion_beta
	    ft = fisher_t(dfn=f4, dfd=f3)
	    if fisher(ft, fp):
	        print("Рівняння регресії є адекватним")
	        break
	    else:
	        print("Рівняння регресії не є адекватним")
	        break

	else:
	    print("Дисперсія не є однорідною")
	    m += 1