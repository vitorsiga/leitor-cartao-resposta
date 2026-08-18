"""
Servidor de leitura de cartao-resposta (OMR) via HTTP.

Reimplementacao EXATA (mesma logica, mesmos parametros) do pipeline que
roda client-side em JavaScript/OpenCV.js no leitor_cartao_resposta.html --
toda a logica foi validada extensivamente contra fotos reais ao longo do
desenvolvimento; aqui so foi portada para Python/OpenCV nativo.

Uso:
    pip install flask flask-cors opencv-python pyzbar numpy
    # pyzbar tambem precisa da lib do sistema: sudo apt install libzbar0
    python3 servidor_omr.py
    # sobe em http://0.0.0.0:5000

Endpoint:
    POST /processar
        form-data:
            foto: arquivo de imagem (jpeg)
            total_questoes: int (10, 20, 30, 40, 50 ou 60)
        retorna JSON no mesmo formato que o cliente client-side gera,
        mais 'bolhas' (coordenadas de cada bolha) e 'imagem_retificada_base64'
        (para o cliente desenhar overlay/miniaturas sem precisar reprocessar)
"""

import time
import base64
import itertools

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from pyzbar.pyzbar import decode as zbar_decode
except ImportError:
    zbar_decode = None  # QR fica indisponivel se pyzbar/libzbar nao estiver instalado

app = Flask(__name__)
CORS(app)  # permite chamadas de qualquer origem (ajuste em producao)

OUT_W, OUT_H = 1700, 3000


# ============================================================================
# 1) CONTRASTE E LIMIAR ADAPTATIVO
# ============================================================================
def melhorar_contraste(gray):
    p_min, p_max = np.percentile(gray, [2, 98])
    if p_max <= p_min:
        return gray
    escala = 255.0 / (p_max - p_min)
    return np.clip((gray.astype(np.float32) - p_min) * escala, 0, 255).astype(np.uint8)


# ============================================================================
# 2) DETECÇÃO DO PAPEL E DOS 4 CANTOS
# ============================================================================
def detectar_papel(gray):
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel_abertura = np.ones((9, 9), np.uint8)
    aberto = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_abertura)
    kernel_fechamento = np.ones((25, 25), np.uint8)
    fechado = cv2.morphologyEx(aberto, cv2.MORPH_CLOSE, kernel_fechamento)
    contours, _ = cv2.findContours(fechado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    maior = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(maior)


REGIOES_METRICA = {
    'top_left': lambda p: p[:, 0] + p[:, 1],
    'top_right': lambda p: -p[:, 0] + p[:, 1],
    'bottom_left': lambda p: p[:, 0] - p[:, 1],
    'bottom_right': lambda p: -p[:, 0] - p[:, 1],
}


def detectar_cantos(img):
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    papel = detectar_papel(gray_original)
    if papel is None:
        return None
    px, py, pw, ph = papel

    gray = melhorar_contraste(gray_original.copy())

    regioes = {
        'top_left': (0.00, 0.15, 0.00, 0.12),
        'top_right': (0.85, 1.00, 0.00, 0.12),
        'bottom_left': (0.00, 0.15, 0.75, 1.00),
        'bottom_right': (0.85, 1.00, 0.75, 1.00),
    }
    area_min = 0.0004 * pw * ph
    cantos = {}

    for nome, (x0f, x1f, y0f, y1f) in regioes.items():
        x0, x1 = px + int(x0f * pw), px + int(x1f * pw)
        y0, y1 = py + int(y0f * ph), py + int(y1f * ph)
        rx0, ry0 = max(0, x0), max(0, y0)
        rx1, ry1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        roi_gray = gray[ry0:ry1, rx0:rx1]
        _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        conts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maior_area, melhor = 0, None
        for c in conts:
            a = cv2.contourArea(c)
            if a > maior_area and a > area_min:
                maior_area, melhor = a, c
        if melhor is not None:
            pontos = melhor.reshape(-1, 2).astype(np.float64)
            valores = REGIOES_METRICA[nome](pontos)
            limiar = np.percentile(valores, 10)
            extremos = pontos[valores <= limiar]
            pf = extremos.mean(axis=0)
            cantos[nome] = (rx0 + pf[0], ry0 + pf[1])

    if len(cantos) != 4:
        return None
    return cantos


def corrigir_perspectiva(img, cantos):
    src = np.float32([cantos['top_left'], cantos['top_right'], cantos['bottom_right'], cantos['bottom_left']])
    dst = np.float32([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (OUT_W, OUT_H))


# ============================================================================
# 3) QR CODE
# ============================================================================
def localizar_regiao_qr(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0, y0 = int(0.55 * w), 0
    x1, y1 = w, int(0.20 * h)
    roi = gray[y0:y1, x0:x1]
    _, thresh = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    dilatado = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    melhor, maior_area = None, 0
    for c in contours:
        a = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh if bh > 0 else 0
        if a > 3000 and 0.7 < aspect < 1.4 and a > maior_area:
            maior_area, melhor = a, (x0 + bx, y0 + by, bw, bh)
    return melhor


def ler_qr(warped):
    """pyzbar (zbar) e nativamente mais robusto que jsQR -- geralmente
    nem precisa do recorte/margem que o cliente precisa tentar varias vezes."""
    if zbar_decode is None:
        return None
    regiao = localizar_regiao_qr(warped)
    if regiao:
        x, y, w, h = regiao
        margem = 40
        x0, y0 = max(0, x - margem), max(0, y - margem)
        x1 = min(warped.shape[1], x + w + margem)
        y1 = min(warped.shape[0], y + h + margem)
        crop = warped[y0:y1, x0:x1]
        resultado = zbar_decode(crop)
        if resultado:
            return resultado[0].data.decode('utf-8', errors='replace')
    # fallback: tenta a imagem inteira
    resultado = zbar_decode(warped)
    return resultado[0].data.decode('utf-8', errors='replace') if resultado else None


# ============================================================================
# 4) MARCADORES DE LINHA (quadrado/traço/losango/3-traços/triângulo)
# ============================================================================
def centroide_poligono(pts):
    n = len(pts)
    A = Cx = Cy = 0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) < 1e-6:
        return np.mean(pts[:, 0]), np.mean(pts[:, 1])
    return Cx / (6 * A), Cy / (6 * A)


def classificar_forma(cnt):
    area = cv2.contourArea(cnt)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    if bw == 0 or bh == 0 or area < 15:
        return None
    fill = area / (bw * bh)
    aspect = bw / bh
    if aspect > 2.2 or aspect < 0.45:
        return 'traco'
    if fill > 0.7 and 0.65 < aspect < 1.5:
        return 'quadrado'
    if 0.3 < fill < 0.68 and 0.65 < aspect < 1.5:
        pts = cnt.reshape(-1, 2).astype(float)
        _, cy = centroide_poligono(pts)
        centro_y = (cy - by) / bh
        return 'triangulo' if centro_y > 0.55 else 'losango'
    return None


FORMA_PARA_POS = {'quadrado': 0, 'losango': 2, 'triangulo': 4}


def posicao_por_marcador(gray, x_coluna_a, y_linha):
    x0, x1 = int(x_coluna_a - 170), int(x_coluna_a - 118)
    y0, y1 = int(y_linha - 22), int(y_linha + 22)
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        return None
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    formas = []
    for c in contours:
        if cv2.contourArea(c) < 15:
            continue
        f = classificar_forma(c)
        if f:
            formas.append(f)
    for f in formas:
        if f in FORMA_PARA_POS:
            return FORMA_PARA_POS[f]
    n_tracos = formas.count('traco')
    if n_tracos >= 2:
        return 3
    if n_tracos == 1:
        return 1
    return None


# ============================================================================
# 5) DETECÇÃO DAS BOLHAS E ATRIBUIÇÃO GENERALIZADA
# ============================================================================
def melhor_subconjunto_de_5(pontos, razao_max=3.0):
    ordenado = sorted(pontos, key=lambda p: p[0])
    n = len(ordenado)
    if n < 5:
        return None
    melhor_razao, melhor_combo = float('inf'), None
    for combo in itertools.combinations(range(n), 5):
        xs = [ordenado[i][0] for i in combo]
        gaps = [xs[i + 1] - xs[i] for i in range(4)]
        if min(gaps) <= 0:
            continue
        razao = max(gaps) / min(gaps)
        if razao < melhor_razao:
            melhor_razao, melhor_combo = razao, combo
    if melhor_combo and melhor_razao <= razao_max:
        return [ordenado[i] for i in melhor_combo]
    return None


def remover_linhas_isoladas(linhas, razao_max=1.8):
    if len(linhas) < 3:
        return linhas
    com_y = sorted([(np.mean([p[1] for p in l]), l) for l in linhas], key=lambda t: t[0])
    ys = [t[0] for t in com_y]
    gaps = np.diff(ys)
    mediana = np.median(gaps)
    resultado = []
    for i, (y, l) in enumerate(com_y):
        gap_ant = ys[i] - ys[i - 1] if i > 0 else float('inf')
        gap_prox = ys[i + 1] - ys[i] if i < len(ys) - 1 else float('inf')
        if min(gap_ant, gap_prox) <= razao_max * mediana:
            resultado.append(l)
    return resultado


def atribuir_questoes_coluna(linhas, gray):
    """Ancora pelo primeiro marcador legivel + espacamento incremental
    (passo a passo, nao distancia direta da 1a linha -- ver historico de
    correcoes: espacamento real varia um pouco ao longo de colunas
    longas e isso acumula erro se calculado de uma vez so)."""
    if not linhas:
        return {}
    com_y = sorted([(l, np.mean([p[1] for p in l]), l[0][0]) for l in linhas], key=lambda t: t[1])

    if len(com_y) == 1:
        l, y, xA = com_y[0]
        pos = posicao_por_marcador(gray, xA, y)
        return {pos: l} if pos is not None else {}

    ys = [t[1] for t in com_y]
    gaps = np.diff(ys)
    espacamento = np.median(gaps)
    if espacamento <= 0:
        return {}

    idx_rel = [0]
    for g in gaps:
        passo = max(1, round(g / espacamento))
        idx_rel.append(idx_rel[-1] + passo)

    ancora = None
    for i, (l, y, xA) in enumerate(com_y):
        pos = posicao_por_marcador(gray, xA, y)
        if ancora is None and pos is not None:
            ancora = pos - idx_rel[i]
    if ancora is None:
        return {}

    resultado = {}
    for i, (l, y, xA) in enumerate(com_y):
        idx_abs = idx_rel[i] + ancora
        if idx_abs >= 0:
            resultado[idx_abs] = l
    return resultado


def gerar_numeros_questao(total):
    por_coluna = total // 2
    return {
        'esquerda': list(range(1, por_coluna + 1)),
        'direita': list(range(por_coluna + 1, total + 1)),
    }


def detectar_bolhas(warped, total_questoes):
    gray_original = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = melhorar_contraste(gray_original.copy())

    regiao_qr = localizar_regiao_qr(warped)
    if regiao_qr:
        x, y, w, h = regiao_qr
        margem = 8
        qx, qy = max(0, x - margem), max(0, y - margem)
        qx1 = min(gray.shape[1], x + w + margem)
        qy1 = min(gray.shape[0], y + h + margem)
        gray[qy:qy1, qx:qx1] = 255

    blur = cv2.medianBlur(gray, 5)
    blocos = {'esquerda': (100, 820), 'direita': (870, 1650)}
    y0, y1 = 150, 2900
    letras = ['A', 'B', 'C', 'D', 'E']
    linhas_por_bloco = {}

    for nome, (x0, x1) in blocos.items():
        roi = blur[y0:y1, x0:x1]
        circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                                     param1=60, param2=18, minRadius=18, maxRadius=35)
        linhas = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            pontos = sorted([(x + x0, y + y0, r) for x, y, r in circles], key=lambda p: p[1])
            tolY = 45
            for p in pontos:
                colocado = False
                for l in linhas:
                    my = np.mean([q[1] for q in l])
                    if abs(my - p[1]) < tolY:
                        l.append(p)
                        colocado = True
                        break
                if not colocado:
                    linhas.append([p])

        linhas_validas = []
        for l in linhas:
            if len(l) < 5 or len(l) > 7:
                continue
            melhor = melhor_subconjunto_de_5(l)
            if melhor:
                linhas_validas.append(melhor)
        linhas_por_bloco[nome] = remover_linhas_isoladas(linhas_validas)

    numeros = gerar_numeros_questao(total_questoes)
    atrib_esq = atribuir_questoes_coluna(linhas_por_bloco['esquerda'], gray)
    atrib_dir = atribuir_questoes_coluna(linhas_por_bloco['direita'], gray)

    grade = {}
    for idx, linha in atrib_esq.items():
        if 0 <= idx < len(numeros['esquerda']):
            q = numeros['esquerda'][idx]
            grade[q] = [{'letra': letras[i], 'x': float(p[0]), 'y': float(p[1]), 'r': float(p[2])}
                        for i, p in enumerate(linha)]
    for idx, linha in atrib_dir.items():
        if 0 <= idx < len(numeros['direita']):
            q = numeros['direita'][idx]
            grade[q] = [{'letra': letras[i], 'x': float(p[0]), 'y': float(p[1]), 'r': float(p[2])}
                        for i, p in enumerate(linha)]

    debug = {
        'totalQuestoes': total_questoes,
        'linhasPorBloco': {
            'esquerda': [round(np.mean([p[1] for p in l])) for l in linhas_por_bloco['esquerda']],
            'direita': [round(np.mean([p[1] for p in l])) for l in linhas_por_bloco['direita']],
        },
    }
    return grade, debug


# ============================================================================
# 6) LEITURA DE PREENCHIMENTO
# ============================================================================
def ler_respostas(warped, grade):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    respostas, ambiguas = {}, set()
    LIMIAR_MARCACAO = 0.28
    MARGEM_AMBIGUA = 0.12

    for q, bolhas in grade.items():
        scores = {}
        for b in bolhas:
            raio_interno = int(b['r'] * 0.65)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(b['x']), int(b['y'])), raio_interno, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            scores[b['letra']] = 1 - mean_val / 255
        letras_ord = sorted(scores.keys(), key=lambda l: -scores[l])
        maior = scores[letras_ord[0]]
        segunda = scores[letras_ord[1]] if len(letras_ord) > 1 else 0
        if maior < LIMIAR_MARCACAO:
            respostas[q] = None
        else:
            respostas[q] = letras_ord[0]
            if (maior - segunda) < MARGEM_AMBIGUA:
                ambiguas.add(str(q))
    return respostas, ambiguas


# ============================================================================
# ENDPOINT HTTP
# ============================================================================
@app.route('/processar', methods=['POST'])
def processar():
    t0 = time.time()

    if 'foto' not in request.files:
        return jsonify({'erro': 'nenhuma foto enviada no campo "foto"'}), 400

    total_questoes = int(request.form.get('total_questoes', 10))

    file_bytes = np.frombuffer(request.files['foto'].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'erro': 'nao foi possivel decodificar a imagem enviada'}), 400

    cantos = detectar_cantos(img)
    if cantos is None:
        return jsonify({
            'erro': 'Não foi possível localizar os 4 marcadores de canto do cartão.',
            'qrcode': None,
            'respostas': [],
            'tempo_processamento_ms': round((time.time() - t0) * 1000),
        }), 200

    warped = corrigir_perspectiva(img, cantos)
    qr_texto = ler_qr(warped)
    grade, debug = detectar_bolhas(warped, total_questoes)
    respostas, ambiguas = ler_respostas(warped, grade)

    respostas_array = [
        {'questao': f'{q:02d}', 'resposta': respostas.get(q), 'ambigua': str(q) in ambiguas}
        for q in range(1, total_questoes + 1)
    ]

    # imagem retificada em base64, para o cliente desenhar overlay/miniaturas
    # sem precisar reenviar/reprocessar nada
    ok, buf = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    imagem_b64 = base64.b64encode(buf).decode() if ok else None

    return jsonify({
        'qrcode': qr_texto,
        'respostas': respostas_array,
        'tempo_processamento_ms': round((time.time() - t0) * 1000),
        '_debug': debug,
        'bolhas': grade,
        'imagem_retificada_base64': f'data:image/jpeg;base64,{imagem_b64}' if imagem_b64 else None,
    })


@app.route('/', methods=['GET'])
def raiz():
    return jsonify({'status': 'ok', 'endpoint': 'POST /processar (form-data: foto, total_questoes)'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
