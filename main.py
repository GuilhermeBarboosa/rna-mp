from re import sub
from tkinter.simpledialog import SimpleDialog
import PySimpleGUI as sg
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use('TkAgg')

sg.theme('Default')


def aproximacao_grafico(a, b, c, d, e):
    entradas = 1
    num_neuronios = int(a)
    taxa_aprendizado = float(b)
    erro_tolerado = 0.001
    listaciclo = []
    listaerro = []
    xmin = -int(c)
    xmin = -xmin
    xmax = int(d)
    npontos = int(e)

    x1 = np.linspace(xmin, xmax, npontos)
    x = np.zeros((npontos, 1))
    for i in range(npontos):
        x[i][0] = x1[i]
    (amostras, vsai) = np.shape(x)

    t1 = (np.sin(x) / 2) * (np.cos(2 * x))
    t = np.zeros((1, amostras))
    for i in range(amostras):
        t[0][i] = t1[i]
    (vsai, amostras) = np.shape(t)
    vanterior = np.zeros((entradas, num_neuronios))
    aleatorio = 1
    for i in range(entradas):
        for j in range(num_neuronios):
            vanterior[i][j] = rd.uniform(-aleatorio, aleatorio)
    v0anterior = np.zeros((1, num_neuronios))
    for j in range(num_neuronios):
        v0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

    wanterior = np.zeros((num_neuronios, vsai))
    aleatorio = 0.2
    for i in range(num_neuronios):
        for j in range(vsai):
            wanterior[i][j] = rd.uniform(-aleatorio, aleatorio)
    w0anterior = np.zeros((1, vsai))
    for j in range(vsai):
        w0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

    vnovo = np.zeros((entradas, num_neuronios))
    v0novo = np.zeros((1, num_neuronios))
    wnovo = np.zeros((num_neuronios, vsai))
    w0novo = np.zeros((1, vsai))

    zin = np.zeros((1, num_neuronios))
    z = np.zeros((1, num_neuronios))

    deltinhak = np.zeros((vsai, 1))
    deltaw0 = np.zeros((vsai, 1))
    deltinha = np.zeros((1, num_neuronios))

    xaux = np.zeros((1, entradas))
    h = np.zeros((vsai, 1))
    target = np.zeros((vsai, 1))
    deltinha2 = np.zeros((num_neuronios, 1))

    ciclo = 0
    errototal = 100000

    while erro_tolerado < errototal and ciclo <= 5000:
        errototal = 0

        for padrao in range(amostras):
            for j in range(num_neuronios):
                zin[0][j] = np.dot(x[padrao, :], vanterior[:, j]) + v0anterior[0][j]

            z = np.tanh(zin)

            yin = np.dot(z, wanterior) + w0anterior

            y = np.tanh(yin)

            for m in range(vsai):
                h[m][0] = y[0][m]

            for m in range(vsai):
                target[m][0] = t[0][padrao]

            errototal = errototal + np.sum(0.5 * ((target - h) ** 2))

            deltinhak = (target - h) * (1 + h) * (1 - h)

            deltaw = taxa_aprendizado * (np.dot(deltinhak, z))
            deltaw0 = taxa_aprendizado * deltinhak
            deltinhain = np.dot(np.transpose(deltinhak), np.transpose(wanterior))

            deltinha = deltinhain * (1 + z) * (1 - z)

            for m in range(num_neuronios):
                deltinha2[m][0] = deltinha[0][m]

            for k in range(entradas):
                xaux[0][k] = x[padrao][k]

            deltav = taxa_aprendizado * np.dot(deltinha2, xaux)
            deltav0 = taxa_aprendizado * deltinha

            vnovo = vanterior + np.transpose(deltav)
            v0novo = v0anterior + np.transpose(deltav0)

            wnovo = wanterior + np.transpose(deltaw)
            w0novo = w0anterior + np.transpose(deltaw0)
            vanterior = vnovo
            v0anterior = v0novo
            wanterior = wnovo
            w0anterior = w0novo
        ciclo = ciclo + 1
        listaciclo.append(ciclo)
        listaerro.append(errototal)
        print("CICLO\t ERRO")
        print(ciclo, '\t', errototal)

        zin2 = np.zeros((1, num_neuronios))
        z2 = np.zeros((1, num_neuronios))
        t2 = np.zeros((amostras, 1))

        for i in range(amostras):
            for j in range(num_neuronios):
                zin2[0][j] = np.dot(x[i, :], vanterior[:, j]) + v0anterior[0][j]

                z2 = np.tanh(zin2)

            yin2 = np.dot(z2, wanterior) + w0anterior
            y2 = np.tanh(yin2)

            t2[i][0] = y2

        update_graph(x, t1, t2)
        update_error(listaciclo, listaerro)

    return ciclo, errototal


def update_window(ciclo, erro):
    window['-CICLOFINAL-'].update(visible=True)
    window['-ERROFINAL-'].update(visible=True)
    window['-CICLOFINAL-'].update(ciclo)
    window['-ERROFINAL-'].update(erro)


def update_graph(x, t1, t2):
    axes = fig.axes
    axes[0].cla()
    axes[0].plot(x, t1, color="red")
    axes[0].plot(x, t2, color="blue")
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack()


def update_error(ciclo, erro):
    axes2 = fig2.axes
    axes2[0].cla()
    axes2[0].plot(ciclo, erro, color="purple")
    figure_canvas_agg2.draw()
    figure_canvas_agg2.get_tk_widget().pack()
    window.refresh()


col1 = [
    [sg.Text('Aproximação de Função', font=('bold', 20), justification='top')],
    [sg.Text('Neurônios: ', font=('normal', 16), size=(30, 1)), sg.InputText()],
    [sg.Text('Alfa: ', font=('normal', 16), size=(30, 1)), sg.InputText()],
    [sg.Text('Valor mínimo do gráfico: ', font=('normal', 16), size=(30, 1)), sg.InputText()],
    [sg.Text('Valor máximo do gráfico: ', font=('normal', 16), size=(30, 1)), sg.InputText()],
    [sg.Text('Número de pontos do gráfico: ', font=('normal', 16), size=(30, 1)), sg.InputText()],
    [sg.ReadButton('Treinar')],
]

col2 = [
    [sg.Text('Gráfico da Função e do Teste', font=('bold', 16), justification='center')],
    [sg.Text('Função Verdadeira em VERMELHO', font=('normal', 10), background_color="red", text_color=('white'))],
    [sg.Text('Aproximação em Azul', font=('normal', 10), background_color="blue", text_color=('white'))],
    [sg.Canvas(size=(200, 150), key='-CANVAS1-')],
]

col3 = [
    [sg.Text('Gráfico do erro total', font=('bold', 16), justification='center')],
    [sg.Text('Erro Quadrático Final: ', font=('bold', 12)), sg.Text(key='-ERROFINAL-', font=('bold', 12))],
    [sg.Text('', font=('normal', 8))],
    [sg.Canvas(size=(200, 150), key='-CANVAS2-')],
]

col4 = [
    [sg.Text('Nº Ciclos: ', font=('bold', 12)), sg.Text(key='-CICLOFINAL-', font=('bold', 12))],
]

layout = [
    [sg.Column(col1)],
    [sg.Column(col2), sg.Column(col3)],
    [sg.Column(col4)]
]

window = sg.Window('Aproximação de Função', layout, finalize=True)

fig = Figure(figsize=(4, 3))
fig.add_subplot(111).plot([], [])
fig.set_facecolor("#f0f0f0")
figure_canvas_agg = FigureCanvasTkAgg(fig, window['-CANVAS1-'].tk_canvas)
figure_canvas_agg.draw()
figure_canvas_agg.get_tk_widget().pack()

fig2 = Figure(figsize=(4, 3))
fig2.add_subplot(111).plot([], [])
fig2.set_facecolor("#f0f0f0")
figure_canvas_agg2 = FigureCanvasTkAgg(fig2, window['-CANVAS2-'].tk_canvas)
figure_canvas_agg2.draw()
figure_canvas_agg2.get_tk_widget().pack()

while True:
    event, values = window.read()
    if event == 'Cancel' or event == sg.WIN_CLOSED:
        break
    if event == 'Treinar':
        x = aproximacao_grafico(values[0], values[1], values[2], values[3], values[4])
        update_window(x[0], x[1])