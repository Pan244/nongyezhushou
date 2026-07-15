"""
农业病虫害知识库 — ChromaDB 向量存储
每个条目包含：作物、病害名称、症状、原因、防治方案、农药配比
"""
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data" / "chroma_db"

# 内置知识库数据
CROP_KNOWLEDGE = [
    # ===== 水稻 =====
    {
        "crop": "水稻", "disease": "稻瘟病",
        "symptoms": "叶片出现梭形病斑，边缘褐色、中央灰白色。严重时整叶枯死，穗颈变褐折断。",
        "cause": "稻瘟病菌（Magnaporthe oryzae），高温高湿、偏施氮肥时易发。",
        "treatment": "1. 用75%三环唑可湿性粉剂 30克/亩 兑水50公斤喷雾。2. 每7-10天一次，连喷2-3次。3. 最后一次施药距收割不少于21天。",
        "prevention": "选用抗病品种、合理施肥（氮磷钾配合）、浅水灌溉。"
    },
    {
        "crop": "水稻", "disease": "纹枯病",
        "symptoms": "叶鞘出现水渍状暗绿色斑，后扩大成云纹状大斑，边缘褐色。严重时叶片枯死。",
        "cause": "纹枯病菌（Rhizoctonia solani），高温高湿、种植过密、氮肥过多时重发。",
        "treatment": "1. 用5%井冈霉素水剂 150毫升/亩 兑水60公斤喷雾。2. 每7天一次，连喷2次。3. 喷药时重点喷植株中下部。",
        "prevention": "合理密植、浅水勤灌、增施磷钾肥。"
    },
    {
        "crop": "水稻", "disease": "稻飞虱",
        "symptoms": "叶片发黄，植株矮小，严重时成片枯死。叶背和茎秆基部有大量小虫。",
        "cause": "褐飞虱、白背飞虱等刺吸式害虫，迁飞性强、繁殖快。",
        "treatment": "1. 用25%吡蚜酮可湿性粉剂 20克/亩 兑水45公斤喷雾。2. 每7-10天一次。3. 最后一次施药距收割不少于14天。",
        "prevention": "选用抗虫品种、保护天敌（蜘蛛、黑肩绿盲蝽）。"
    },

    # ===== 小麦 =====
    {
        "crop": "小麦", "disease": "赤霉病",
        "symptoms": "穗部出现粉红色霉层，籽粒干瘪发红。严重时整穗枯白，籽粒含毒素。",
        "cause": "赤霉病菌（Fusarium graminearum），抽穗扬花期遇连续阴雨天气易爆发。",
        "treatment": "1. 用25%氰烯菌酯悬浮剂 100毫升/亩 兑水30公斤喷雾。2. 在抽穗扬花初期喷第一次，5-7天后喷第二次。3. 最后一次施药距收割不少于28天。",
        "prevention": "选用抗病品种、合理轮作、及时清除病残体。"
    },
    {
        "crop": "小麦", "disease": "锈病",
        "symptoms": "叶片和茎秆上出现铁锈色粉末状孢子堆，严重时叶片枯黄早衰。",
        "cause": "锈病菌（Puccinia），气流传播，适温15-25°C，湿度大时传播快。",
        "treatment": "1. 用25%三唑酮可湿性粉剂 50克/亩 兑水50公斤喷雾。2. 每10-15天一次。3. 最后一次施药距收割不少于20天。",
        "prevention": "种植抗病品种、合理施肥、雨后排水。"
    },
    {
        "crop": "小麦", "disease": "蚜虫",
        "symptoms": "叶片卷曲发黄，生长点受损，传播病毒病。穗部受害后籽粒不饱满。",
        "cause": "麦蚜（麦长管蚜、麦二叉蚜等），干旱年份发生重。",
        "treatment": "1. 用10%吡虫啉可湿性粉剂 15克/亩 兑水30公斤喷雾。2. 每7天一次。3. 花期避免喷药。4. 最后一次施药距收割不少于14天。",
        "prevention": "保护瓢虫等天敌、合理灌溉。"
    },

    # ===== 玉米 =====
    {
        "crop": "玉米", "disease": "大斑病",
        "symptoms": "叶片出现长梭形灰褐色大斑，严重时叶片枯焦，影响灌浆。",
        "cause": "大斑病菌（Exserohilum turcicum），温度20-25°C、多雨多雾时流行。",
        "treatment": "1. 用50%多菌灵可湿性粉剂 500倍液喷雾。2. 每7-10天一次，连喷2-3次。3. 最后一次施药距采收不少于21天。",
        "prevention": "选用抗病品种、合理密植、收获后清除病残体。"
    },
    {
        "crop": "玉米", "disease": "玉米螟",
        "symptoms": "叶片出现排孔状咬痕，茎秆蛀孔有虫粪，雄穗折断，雌穗籽粒被害。",
        "cause": "亚洲玉米螟（Ostrinia furnacalis），幼虫蛀食为害。",
        "treatment": "1. 在大喇叭口期用1.5%辛硫磷颗粒剂 1公斤/亩 拌细土撒施于心叶。2. 穗期用Bt可湿性粉剂800倍液喷穗。3. 最后一次施药距采收不少于15天。",
        "prevention": "收获后处理秸秆消灭越冬虫源、释放赤眼蜂。"
    },
    {
        "crop": "玉米", "disease": "草地贪夜蛾",
        "symptoms": "幼虫取食叶片形成窗格状或大孔洞，严重时吃光叶片仅剩叶脉。也蛀食心叶和果穗。",
        "cause": "草地贪夜蛾（Spodoptera frugiperda），迁飞性强，繁殖快。",
        "treatment": "1. 用5%甲氨基阿维菌素苯甲酸盐微乳剂 10毫升/亩 兑水30公斤喷雾。2. 低龄幼虫期防治最佳。3. 每7-10天一次。4. 最后一次施药距采收不少于14天。",
        "prevention": "性诱剂监测诱杀、种植早熟品种避开危害高峰。"
    },

    # ===== 棉花 =====
    {
        "crop": "棉花", "disease": "棉铃虫",
        "symptoms": "幼虫蛀食蕾、花、铃，造成脱落和烂铃。一头幼虫可危害10-20个蕾铃。",
        "cause": "棉铃虫（Helicoverpa armigera），高温干旱年份重发。",
        "treatment": "1. 用2.5%高效氯氟氰菊酯乳油 25毫升/亩 兑水45公斤喷雾。2. 卵孵化盛期至低龄幼虫期用药。3. 每7天一次。4. 最后一次施药距采收不少于14天。",
        "prevention": "种植转基因抗虫棉、秋耕冬灌灭蛹、频振式杀虫灯诱杀。"
    },
    {
        "crop": "棉花", "disease": "枯萎病",
        "symptoms": "叶片萎蔫下垂，似缺水状。剖开茎秆可见维管束变褐色。严重时整株枯死。",
        "cause": "枯萎病菌（Fusarium oxysporum），土传病害，连作地发病重。",
        "treatment": "1. 发病初期用50%多菌灵可湿性粉剂 800倍液灌根。2. 每株灌0.5升药液。3. 每7-10天灌一次，连灌2-3次。",
        "prevention": "与禾本科作物轮作3年以上、选用抗病品种。"
    },

    # ===== 蔬菜 =====
    {
        "crop": "番茄", "disease": "晚疫病",
        "symptoms": "叶片出现水渍状暗绿色斑，湿度大时叶背有白色霉层。果实上出现暗褐色斑块。",
        "cause": "晚疫病菌（Phytophthora infestans），低温高湿（温度18-22°C、湿度>90%）时暴发。",
        "treatment": "1. 用72%霜脲·锰锌可湿性粉剂 800倍液喷雾。2. 每5-7天一次，连喷3-4次。3. 最后一次施药距采收不少于7天。",
        "prevention": "选用抗病品种、加强通风降湿、高畦栽培。"
    },
    {
        "crop": "番茄", "disease": "白粉虱",
        "symptoms": "叶片发黄、卷曲，叶背有大量白色小飞虫。分泌蜜露引发煤污病。",
        "cause": "温室白粉虱（Trialeurodes vaporariorum），温室大棚内终年发生。",
        "treatment": "1. 用25%噻虫嗪水分散粒剂 3000倍液喷雾。2. 每7天一次，连喷2-3次。3. 重点喷叶背。4. 最后一次施药距采收不少于5天。",
        "prevention": "悬挂黄色粘虫板、释放丽蚜小蜂。"
    },
    {
        "crop": "黄瓜", "disease": "霜霉病",
        "symptoms": "叶片出现多角形黄色斑，叶背有紫灰色霉层。严重时叶片枯焦。",
        "cause": "霜霉病菌（Pseudoperonospora cubensis），适温16-22°C、多雨多雾时流行。",
        "treatment": "1. 用58%甲霜·锰锌可湿性粉剂 600倍液喷雾。2. 每5-7天一次，连喷3次。3. 最后一次施药距采收不少于3天。",
        "prevention": "选用抗病品种、加强通风、地膜覆盖降低湿度。"
    },
    {
        "crop": "白菜", "disease": "软腐病",
        "symptoms": "外叶萎蔫，叶球基部腐烂发臭，有黏稠菌脓。严重时整株腐烂倒塌。",
        "cause": "软腐病菌（Pectobacterium carotovorum），通过伤口侵入，高温高湿时流行。",
        "treatment": "1. 发病初期用72%农用链霉素可溶性粉剂 4000倍液喷淋。2. 重点喷淋植株基部。3. 每7天一次。4. 最后一次施药距采收不少于7天。",
        "prevention": "高畦栽培防积水、及时防治地下害虫减少伤口。"
    },

    # ===== 果树 =====
    {
        "crop": "苹果", "disease": "腐烂病",
        "symptoms": "树干皮层出现红褐色坏死斑，有酒糟味，病部干缩凹陷。严重时整株枯死。",
        "cause": "腐烂病菌（Valsa mali），树势衰弱、冻害后易发。",
        "treatment": "1. 刮除病斑至健康皮层，涂50%多菌灵可湿性粉剂 50倍液。2. 或涂843康复剂原液。3. 春、秋季各处理一次。",
        "prevention": "加强肥水管理增强树势、冬季树干涂白防冻。"
    },
    {
        "crop": "柑橘", "disease": "红蜘蛛",
        "symptoms": "叶片出现密集黄白色小点，严重时叶片灰白无光泽。叶背有细小红色虫体。",
        "cause": "柑橘全爪螨（Panonychus citri），高温干旱季节爆发。",
        "treatment": "1. 用1.8%阿维菌素乳油 3000倍液喷雾。2. 每7-10天一次。3. 轮换使用不同药剂防止抗性。4. 最后一次施药距采收不少于14天。",
        "prevention": "保护捕食螨等天敌、合理灌溉增加果园湿度。"
    },

    # ===== 常见缺素症 =====
    {
        "crop": "通用", "disease": "缺氮",
        "symptoms": "植株矮小，叶片从老叶开始均匀黄化，严重时全株枯黄。",
        "cause": "土壤氮素供应不足，或根系吸收障碍。",
        "treatment": "追施尿素 10-15公斤/亩，或叶面喷施1%尿素溶液。每7天一次，连喷2次。",
        "prevention": "基肥施足有机肥、测土配方施肥。"
    },
    {
        "crop": "通用", "disease": "缺钾",
        "symptoms": "老叶叶尖和叶缘开始黄化焦枯，像火烧一样。叶片卷曲，茎秆软弱。",
        "cause": "土壤有效钾含量低，或沙质土钾流失严重。",
        "treatment": "追施硫酸钾 10-15公斤/亩，或叶面喷施0.3%磷酸二氢钾溶液。每7天一次，连喷2-3次。",
        "prevention": "增施有机肥和草木灰、秸秆还田。"
    },
    {
        "crop": "通用", "disease": "缺铁",
        "symptoms": "新叶叶脉间失绿黄化，叶脉仍为绿色，形成网状花纹。严重时新叶全白。",
        "cause": "碱性土壤铁元素被固定，或根系受损影响铁吸收。",
        "treatment": "叶面喷施0.2%硫酸亚铁溶液（加少量柠檬酸），每5-7天一次，连喷2-3次。",
        "prevention": "增施有机肥改良土壤、避免过量施用磷肥。"
    },
]


def _get_embedding_fn():
    """获取 Embedding 函数，远程环境降级到简易模式"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    # 尝试用 OpenAI 兼容的 Embedding
    if api_key:
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small",
            )
            ef(["test"])  # 验证能调通
            return ef
        except Exception:
            pass
    # 回退：简易 Embedding（不依赖外部模型，Day 06 教学内容）
    return SimpleEmbeddingFunction()


class SimpleEmbeddingFunction:
    """简易 Embedding — 课程 Day 06 教学实现，不依赖任何外部模型"""
    def __init__(self):
        import hashlib
        self._h = hashlib

    def __call__(self, texts):
        import math
        out = []
        for t in texts:
            b = self._h.sha256(t.encode()).digest()
            v = [(b[i % len(b)] / 255.0) * 2 - 1 for i in range(128)]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v] if n else v)
        return out


def build_knowledge_base():
    """构建/重建病虫害知识库"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    DB_DIR.mkdir(parents=True, exist_ok=True)
    ef = _get_embedding_fn()

    try: client.delete_collection("crop_diseases")
    except Exception: pass

    col = client.create_collection(
        name="crop_diseases",
        embedding_function=ef,
        metadata={"description": "农作物病虫害知识库", "hnsw:space": "cosine"},
    )

    docs = []
    metas = []
    ids = []
    for i, item in enumerate(CROP_KNOWLEDGE):
        text = f"【{item['crop']}】{item['disease']}\n症状：{item['symptoms']}\n原因：{item['cause']}\n防治：{item['treatment']}\n预防：{item['prevention']}"
        docs.append(text)
        metas.append({"crop": item["crop"], "disease": item["disease"]})
        ids.append(f"disease_{i}")

    col.add(documents=docs, metadatas=metas, ids=ids)
    print(f"✅ 病虫害知识库已就绪：{len(docs)} 条记录（{'简易Embedding' if isinstance(ef, SimpleEmbeddingFunction) else 'OpenAI Embedding'}）")
    return col


def get_collection():
    """获取知识库 Collection"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        return client.get_collection("crop_diseases")
    except Exception:
        return build_knowledge_base()


def search_knowledge(query: str, n_results: int = 3) -> list:
    """搜索病虫害知识"""
    try:
        col = get_collection()
        results = col.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(1 - results["distances"][0][i], 3),
            })
        return out
    except Exception as e:
        print(f"知识库搜索失败: {e}")
        return []
