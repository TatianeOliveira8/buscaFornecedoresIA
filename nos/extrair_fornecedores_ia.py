import json
import os
import re
import requests
from typing import Dict, Any, List
from html import unescape
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from estado import EstadoReposicao

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("API_KEY"),
    temperature=0,
)


def investigar_site(url: str, nome_produto: str) -> Dict[str, Any]:
    resultado = {
        "pagina_consultada": False,
        "produtos_identificados": [],
        "contatos": [],
        "informacoes_disponiveis": [],
        "evidencia_textual": "",
    }

    if not url or "google.com/maps" in url:
        return resultado

    try:
        resposta = requests.get(
            url,
            headers={"User-Agent": "BuscaFornecedores/1.0"},
            timeout=5,
        )
        resposta.raise_for_status()
    except requests.RequestException:
        return resultado

    resultado["pagina_consultada"] = True
    texto = unescape(re.sub(r"<[^>]+>", " ", resposta.text))
    texto = re.sub(r"\s+", " ", texto).strip()
    texto_lower = texto.lower()

    emails = sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", texto)))
    telefones = sorted(set(re.findall(r"(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}[-\s]?\d{4}", texto)))
    resultado["contatos"] = emails + telefones

    produto_limpo = nome_produto.strip()
    if produto_limpo and produto_limpo.lower() in texto_lower:
        resultado["produtos_identificados"].append(produto_limpo)

    if resultado["produtos_identificados"]:
        resultado["informacoes_disponiveis"].append("produto mencionado na página")
    if resultado["contatos"]:
        resultado["informacoes_disponiveis"].append("meio de contato encontrado")
    resultado["evidencia_textual"] = texto[:1200]

    return resultado


def avaliar_compatibilidade(
    fornecedor: Dict[str, Any],
    nome_produto: str,
    evidencia_adicional: str = "",
) -> Dict[str, Any]:
    investigacao = fornecedor.get("investigacao", {})
    contexto = (
        f"Fornecedor: {fornecedor.get('nome_fornecedor')}\n"
        f"Produto solicitado: {nome_produto}\n"
        f"Página consultada: {investigacao.get('pagina_consultada')}\n"
        f"Produtos identificados por correspondência literal: "
        f"{investigacao.get('produtos_identificados', [])}\n"
        f"Trecho da página: {investigacao.get('evidencia_textual', '')}\n"
        f"Resultado de pesquisa adicional: {evidencia_adicional or 'não realizada'}"
    )
    prompt = f"""
Analise somente as evidências abaixo. Não invente disponibilidade, preço ou produtos.
Responda exclusivamente em JSON válido com as chaves:
compatibilidade (confirmada, provavel ou nao_confirmada),
precisa_pesquisa_adicional (true ou false),
justificativa (uma frase baseada nas evidências).

{contexto}
"""
    try:
        resposta = llm.invoke(prompt).content.strip()
        inicio = resposta.find("{")
        fim = resposta.rfind("}")
        dados = json.loads(resposta[inicio:fim + 1])
        compatibilidade = dados.get("compatibilidade", "nao_confirmada")
        if compatibilidade not in {"confirmada", "provavel", "nao_confirmada"}:
            compatibilidade = "nao_confirmada"
        return {
            "compatibilidade": compatibilidade,
            "precisa_pesquisa_adicional": bool(dados.get("precisa_pesquisa_adicional")),
            "justificativa": str(dados.get("justificativa", "Evidência insuficiente.")),
        }
    except (ChatGoogleGenerativeAIError, ValueError, TypeError, json.JSONDecodeError) as e:
        return {
            "compatibilidade": "nao_confirmada",
            "precisa_pesquisa_adicional": True,
            "justificativa": f"A análise estruturada não foi confirmada: {e}",
        }


def pesquisar_adicionalmente(
    fornecedor: Dict[str, Any],
    nome_produto: str,
    localizacao: str,
) -> str:
    chave = os.getenv("SERPER_API_KEY", "")
    if not chave:
        return ""
    try:
        resposta = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": chave, "Content-Type": "application/json"},
            json={"q": f'"{nome_produto}" "{fornecedor["nome_fornecedor"]}" {localizacao}', "gl": "br", "hl": "pt-br"},
            timeout=8,
        )
        resposta.raise_for_status()
        resultados = resposta.json().get("organic", [])[:3]
        return " ".join(
            f"{item.get('title', '')}: {item.get('snippet', '')}"
            for item in resultados
        )[:2000]
    except (requests.RequestException, ValueError):
        return ""


def avaliar_fornecedores(estado: EstadoReposicao) -> Dict[str, Any]:
    fornecedores = estado.get("fornecedores_dentro_raio", [])
    nome_produto = estado["nome_produto"]
    localizacao = estado.get("localizacao_usuario", "")
    avisos = list(estado.get("avisos", []))

    for fornecedor in fornecedores:
        avaliacao = avaliar_compatibilidade(fornecedor, nome_produto)
        if avaliacao["precisa_pesquisa_adicional"]:
            adicional = pesquisar_adicionalmente(fornecedor, nome_produto, localizacao)
            if adicional:
                avaliacao = avaliar_compatibilidade(fornecedor, nome_produto, adicional)
                fornecedor["investigacao"]["pesquisa_adicional"] = adicional
            fornecedor["investigacao"]["pesquisa_adicional_realizada"] = bool(adicional)
        fornecedor["investigacao"]["compatibilidade"] = avaliacao["compatibilidade"]
        fornecedor["investigacao"]["justificativa"] = avaliacao["justificativa"]
        fornecedor["investigacao"]["evidencia_confirmada"] = (
            avaliacao["compatibilidade"] == "confirmada"
        )

    return {"fornecedores_dentro_raio": fornecedores, "avisos": avisos}


def extrair_fornecedores_ia(estado: EstadoReposicao) -> Dict[str, Any]:
    nome_produto = estado["nome_produto"]
    texto_web = estado.get("resultados_web_brutos", "")
    localizacao = estado.get("localizacao_usuario", "").strip() or "São Paulo - SP"
    avisos = list(estado.get("avisos", []))

    lugares_google = []
    try:
        if texto_web and texto_web.strip().startswith("["):
            lugares_google = json.loads(texto_web)
    except Exception:
        lugares_google = []

    fornecedores_base = []
    for idx, p in enumerate(lugares_google[:5], start=1):
        nome = p.get("title", f"Distribuidora {nome_produto}")
        endereco = p.get("address", localizacao)
        lat = p.get("latitude")
        lon = p.get("longitude")
        site = p.get("website")
        link = site or f"https://www.google.com/maps/search/?api=1&query={requests.utils.quote(nome + ' ' + localizacao)}"

        cidade_forn = localizacao
        if "," in endereco:
            partes = [pt.strip() for pt in endereco.split(",")]
            if len(partes) >= 2:
                cidade_forn = partes[-2]

        try:
            posicao = int(p.get("position", idx))
        except (TypeError, ValueError):
            posicao = idx

        fornecedores_base.append({
            "id_fornecedor": idx,
            "nome_fornecedor": nome,
            "endereco_completo": endereco,
            "latitude": lat,
            "longitude": lon,
            "cidade": cidade_forn,
            "link": link,
            "relevancia": max(1, 6 - posicao),
            "nome_produto": nome_produto,
        })

    for fornecedor in fornecedores_base:
        fornecedor["investigacao"] = investigar_site(
            fornecedor.get("link", ""),
            nome_produto,
        )

    if not fornecedores_base:
        avisos.append(
            f"Nenhum fornecedor real foi retornado pelo Serper para '{localizacao}'."
        )

    return {
        "fornecedores_encontrados": fornecedores_base,
        "avisos": avisos,
    }
