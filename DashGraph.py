import io
import base64
import dash
from dash import dcc, html
import matplotlib
import matplotlib.pyplot as plt
from pylab import arange
import numpy as np  # Importe numpy para gerar cores únicas

matplotlib.use('Agg')

pontos = []
cores_pontos = []  
mensagens = []

app = dash.Dash(__name__)
server = app.server

form_style = {
    'border': '2px solid #2E8B57',
    'border-radius': '10px',
    'padding': '20px',
    'background-color': '#F0F8FF',  # Alice Blue
    'margin': '20px',
    'font-family': 'Arial, sans-serif',
    'justify-content': 'center'
}

label_style = {
    'margin-right': '10px',
    'font-size': '16px',
    'text-align': 'center'
}

input_style = {
    'margin-right': '10px',
    'padding': '5px',
    'font-size': '16px'
}

button_style = {
    'background-color': '#2E8B57',
    'color': 'white',
    'border': '2px solid #2E8B57',
    'border-radius': '5px',
    'padding': '10px 20px',
    'font-size': '16px',
    'cursor': 'pointer',
    'margin-top': '15px',
    'margin-left': '15px',
}

app.layout = html.Div(style={'background-color': '#E6E6FA'}, children=[
    html.H1('Proporção de gases GEE e produção de sucata', style={'color': '#2E8B57', 'text-align': 'center'}),
    html.Div(style=form_style, children=[
        html.P('Favor utilizar PONTO no lugar da VÍRGULA',
               style={'font-weight': 'bold', 'font-size': '14px', 'text-align': 'center', 'margin-bottom': '10px'}),
        html.Label('Emissão de Sucata em decimal', style=label_style),
        dcc.Input(id='input-x', type='number', value=0.0, style=input_style),
        html.Label('Emissão de GEE em decimal', style=label_style),
        dcc.Input(id='input-y', type='number', value=0.0, style=input_style),
        html.Button('Calcular', id='button', style=button_style),
        html.Button('Limpar', id='limpar', style=button_style),
    ]),
    html.Div([
        html.Div(id='message-container', style={'flex': '1', 'text-align': 'left', 'margin-left': '20px'}),
        html.Div(id='graph-container', style={'flex': '1', 'margin-left': '20px'})
    ], style={'display': 'flex', 'margin-bottom': '20px'})
])

@app.callback(
    [dash.dependencies.Output('message-container', 'children'),
     dash.dependencies.Output('graph-container', 'children'),
     dash.dependencies.Output('input-x', 'value'),
     dash.dependencies.Output('input-y', 'value')],
    [dash.dependencies.Input('button', 'n_clicks'),
     dash.dependencies.Input('limpar', 'n_clicks')],
    [dash.dependencies.State('input-x', 'value'),
     dash.dependencies.State('input-y', 'value')]
)
def update_output(n_clicks_button, n_clicks_limpar, x, y):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'button':
        if n_clicks_button is None:
            return '', [html.Img(src=gerar_grafico([], []))], x, y
        else:
            X = float(x)
            Y = float(y)

            pontos.append((X, Y))
            # Gerar uma cor única para este ponto
            cor = '#%02X%02X%02X' % tuple(np.random.choice(range(256), size=3))
            cores_pontos.append(cor)

            #CALCULA QUANTO FALTA PARA CHEGAR AS THRESHOLDE GEE
            Thresholders = []

            Th1 = (2.8 - (2.45*X)) - Y
            Thresholders.insert(0, Th1)
            Th1 = round(Th1, 2)

            Th2 = (2.0 - (1.75*X)) - Y
            Thresholders.insert(0, Th2)
            Th2 = round(Th2, 2)

            Th3 = (1.2 - (1.05*X)) - Y
            Thresholders.insert(0, Th3)
            Th3 = round(Th3, 2)

            Th4 = (0.4 - (0.35*X)) - Y
            Thresholders.insert(0, Th4)
            Th4 = round(Th4, 2)

            #CALCULA QUANTO FALTA PARA CHEGAR A THRESHOLDE SUCATA
            Sucata = []

            Su1 = (X - ((2.8 - Y) /2.45))
            Sucata.insert(0, Su1)
            Su1 = round(Su1, 2)

            Su2 = (X - ((2.0 - Y) /1.75))
            Sucata.insert(0, Su2)
            Su2 = round(Su2, 2)

            Su3 = (X - ((1.2 - Y) /1.05))
            Sucata.insert(0, Su3)
            Su3 = round(Su3, 2)

            Su4 = (X - ((0.4 - Y) /0.35))
            Sucata.insert(0, Su4)
            Su4 = round(Su4, 2)

            #MENSAGENS NA TELA
            mensagem = []
            if Th1 < 0:
                mensagem.append(html.Div([
                    html.Strong(f'Ponto {len(pontos)}:'), f'({X}, {Y}) \nPara alcançar a Thresholde 1 deve diminuir as emissões de GEE em {round(abs(Th1),2)} ou então deve diminuir a produção de sucata em {round(abs(Su1),2)}'
                ], style={'background-color': cor}))  
            if Th2 < 0:
                mensagem.append(html.Div([
                    html.Strong(f'Ponto {len(pontos)}:'), f'({X}, {Y}) \nPara alcançar a Thresholde 2 deve diminuir as emissões de GEE em {round(abs(Th2),2)} ou então deve diminuir a produção de sucata em {round(abs(Su2),2)}'
                ], style={'background-color': cor}))  
            if Th3 < 0:
                mensagem.append(html.Div([
                    html.Strong(f'Ponto {len(pontos)}:'), f'({X}, {Y}) \nPara alcançar a Thresholde 3 deve diminuir as emissões de GEE em {round(abs(Th3),2)} ou então deve diminuir a produção de sucata em {round(abs(Su3),2)}'
                ], style={'background-color': cor}))  
            if Th4 < 0:
                mensagem.append(html.Div([
                    html.Strong(f'Ponto {len(pontos)}:'), f'({X}, {Y}) \nPara alcançar a Thresholde 4 deve diminuir as emissões de GEE em {round(abs(Th4),2)} ou então deve diminuir a produção de sucata em {round(abs(Su4),2)}'
                ], style={'background-color': cor}))  

            mensagens.extend(mensagem)
            #mensagens.append('')

            return mensagem_html(), graph_html(), x, y
    elif button_id == 'limpar':
        if n_clicks_limpar is not None:
            pontos.clear()
            cores_pontos.clear()  
            mensagens.clear()
            x, y = 0.0, 0.0
            return '', [html.Img(src=gerar_grafico([], []))], x, y

    return '', [html.Img(src=gerar_grafico(pontos, cores_pontos))], x, y

def gerar_grafico(pontos, cores_pontos):
    fig, ax = plt.subplots()

    x = arange(0, 1, 0.1)
    ax.plot(x, f1(x), color='red')
    ax.plot(x, f2(x), color='yellow')
    ax.plot(x, f3(x), color='green')
    ax.plot(x, f4(x), color='blue')

    # Marcação dos pontos inseridos
    for i, ponto in enumerate(pontos):
        X, Y = ponto
        cor = cores_pontos[i] if cores_pontos else 'black'
        ax.plot(X, Y, '*', color=cor)  # Use a cor do ponto correspondente
        ax.annotate(f'{i+1}', xy=(X, Y), xytext=(X + 0.02, Y + 0.02), font='Arial, 12', color='black')

    ax.legend(['Thresholder 1', 'Thresholder 2', 'Thresholder 3', 'Thresholder 4'])

    ax.set_title('Crude Steel GHG\nEmissions Intensity')
    ax.set_xlabel('Scrape share of metallica input (proportion)')
    ax.set_ylabel('GHG emissions intensity of crude steel')
    ax.grid()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    encoded_image = base64.b64encode(buffer.read()).decode()

    return 'data:image/png;base64,{}'.format(encoded_image)

def f1(x):
    return 2.8 - (2.45 * x)

def f2(x):
    return 2.0 - (1.75 * x)

def f3(x):
    return 1.2 - (1.05 * x)

def f4(x):
    return 0.4 - (0.35 * x)

def mensagem_html():
    if mensagens and cores_pontos:
        return html.Div(
            [
                html.Div(
                    mensagem,
                    key=f'mensagem-{i}',  
                    style={
                        'font-size': '16px',
                        'padding': '10px',
                        'margin-bottom': '5px',
                        'background-color': cores_pontos[i] if i < len(cores_pontos) else 'white',  # Verifique se i é menor que o comprimento de cores_pontos
                        'box-shadow': '0px 2px 4px rgba(0, 0, 0, 0.1)'
                    }
                ) for i, mensagem in enumerate(mensagens)  # Percorra todas as mensagens
            ],
            style={
                'border': '2px solid #2E8B57',
                'border-radius': '5px',
                'padding': '5px'
            }
        )
    else:
        return html.Div(style={'display': 'none'})

def graph_html():
    return [
        html.Img(src=gerar_grafico(pontos, cores_pontos))
    ]

if __name__ == '__main__':
    app.run_server(debug=True)


