from __future__ import annotations

import csv
import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARS_ROOT = ROOT / "mars_stack"
OUT_ROOT = ROOT / "release_packages" / "mars_field_html_delivery_v2_2_final"
ASSETS_ROOT = OUT_ROOT / "assets"
IMG_ROOT = ASSETS_ROOT / "images"
DOWNLOAD_ROOT = OUT_ROOT / "downloads"
ZIP_PATH = ROOT / "release_packages" / "mars_field_html_delivery_v2_2_final.zip"


def ensure_dirs() -> None:
    paths = [
        OUT_ROOT,
        ASSETS_ROOT,
        IMG_ROOT / "paper",
        IMG_ROOT / "topic",
        IMG_ROOT / "structure",
        DOWNLOAD_ROOT,
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.relative_to(OUT_ROOT).as_posix()


def html_escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_construct_rows(path: Path, phase: str, purpose_map: dict[str, str], status: str) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    output: list[dict[str, str]] = []
    for row in rows:
        name = str(row["name"])
        output.append(
            {
                "name": name,
                "phase": phase,
                "status": status,
                "purpose": purpose_map.get(name, ""),
                "length": str(row.get("length", "")),
                "notes": f"net_charge≈{row.get('net_charge_pH7_approx', '')}, gravy={row.get('gravy', '')}",
            }
        )
    return output


def render_stats(stats: list[dict[str, str]]) -> str:
    blocks = []
    for item in stats:
        blocks.append(
            f"""
            <div class="stat-card">
              <div class="stat-value">{html_escape(item['value'])}</div>
              <div class="stat-label">{html_escape(item['label'])}</div>
              <div class="stat-note">{html_escape(item['note'])}</div>
            </div>
            """
        )
    return "\n".join(blocks)


def render_cards(items: list[dict[str, str]]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            f"""
            <article class="info-card">
              <div class="info-chip">{html_escape(item['tag'])}</div>
              <h3>{html_escape(item['title'])}</h3>
              <p>{html_escape(item['body'])}</p>
              <div class="info-meta">{html_escape(item['meta'])}</div>
            </article>
            """
        )
    return "\n".join(blocks)


def render_table(columns: list[tuple[str, str]], rows: list[dict[str, str]], cls: str = "data-table") -> str:
    thead = "".join(f"<th>{html_escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key, _ in columns)
        body_rows.append(f"<tr>{tds}</tr>")
    return f"""
    <div class="table-wrap">
      <table class="{cls}">
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def render_figure_cards(figures: list[dict[str, str]], extra_class: str = "") -> str:
    blocks = []
    for item in figures:
        blocks.append(
            f"""
            <figure class="figure-card {extra_class}">
              <div class="image-frame">
                <img src="{html_escape(item['src'])}" alt="{html_escape(item['title'])}" loading="lazy" />
              </div>
              <figcaption>
                <div class="figure-label">{html_escape(item['label'])}</div>
                <h3>{html_escape(item['title'])}</h3>
                <p>{html_escape(item['caption'])}</p>
              </figcaption>
            </figure>
            """
        )
    return "\n".join(blocks)


def render_downloads(items: list[dict[str, str]]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            f"""
            <a class="download-card" href="{html_escape(item['href'])}">
              <div class="download-type">{html_escape(item['type'])}</div>
              <div class="download-name">{html_escape(item['name'])}</div>
              <div class="download-note">{html_escape(item['note'])}</div>
            </a>
            """
        )
    return "\n".join(blocks)


def render_command_cards(items: list[dict[str, str]]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            f"""
            <article class="command-card">
              <div class="info-chip">{html_escape(item['tag'])}</div>
              <h3>{html_escape(item['title'])}</h3>
              <pre class="command-block">{html_escape(item['command'])}</pre>
              <p>{html_escape(item['desc'])}</p>
              <div class="info-meta">{html_escape(item['meta'])}</div>
            </article>
            """
        )
    return "\n".join(blocks)


def build_mainline_tables() -> dict[str, list[dict[str, str]]]:
    ares_seed = load_construct_rows(
        ROOT / "designs" / "ares_seed_constructs.scores.csv",
        phase="Ares Round 1",
        status="已产出首轮原型",
        purpose_map={
            "AresC-00_core_only": "稳定 core 对照底盘，用来判断后续 tail 是否拖坏骨架。",
            "AresR-01_N_DNA_tail": "N 端 DNA-tail 方案，验证正电 tail 接近 DNA 的初始概念。",
            "AresR-02_C_DNA_tail": "C 端 DNA-tail 方案，与 N 端挂载方式做构型比较。",
            "AresG-01_N_gel_tail": "N 端 gel-tail 方案，偏向可逆弱装配和保护微环境。",
            "AresG-02_C_gel_tail": "C 端 gel-tail 方案，是当前更值得延伸的保护模块方向。",
            "AresRG-01_N_DNA_C_gel": "DNA-tail 与 gel-tail 双功能拼接方案，做概念上限测试。",
            "AresRG-02_dual_longer_linkers": "更长 linker 的双功能方案，用来测试极限柔性与可表达性的平衡。",
        },
    )
    ares_g2 = load_construct_rows(
        ROOT / "designs" / "aresg_round2_constructs.scores.csv",
        phase="AresG Round 2",
        status="已完成计算设计，待 AF3 与保护实验验证",
        purpose_map={
            "AresG2-01_shortlink_shorttail": "保守版，测试更短 linker 和 tail 是否提升紧凑性与表达友好性。",
            "AresG2-02_midlink_balanced": "平衡版，兼顾 core 稳定和应激弱装配潜力。",
            "AresG2-03_longlink_longtail": "激进版，测试更强的可逆装配与保护趋势。",
        },
    )
    drwh_anc = load_construct_rows(
        ROOT / "designs" / "drwh_asr_ancestor_candidates.scores.csv",
        phase="DrwH ASR Round 1",
        status="已产出祖先面板，待 AF3 与表达友好性筛选",
        purpose_map={
            "DrwHAnc-01_local_near_Node046": "近祖先版本，尽量贴近天然母体，作为保守保护帽对照。",
            "DrwHAnc-02_local_mid_Node054": "中祖先版本，在稳定性与保守性之间找平衡。",
            "DrwHAnc-03_deep_ingroup_Node068": "深祖先版本，测试更共识化骨架是否更适合做 cargo-cap。",
        },
    )
    ares_w = load_construct_rows(
        ROOT / "designs" / "aresw_cargo_cap_constructs.scores.csv",
        phase="AresW cargo-cap",
        status="已设计，待 AF3 与后续拼接验证",
        purpose_map={
            "AresW-00_core_only": "core-only 对照，验证 cap 是否会拖塌基础骨架。",
            "AresW-01_N_cap_native": "天然 DrwH 域 N 端挂载方案。",
            "AresW-02_C_cap_native": "天然 DrwH 域 C 端挂载方案。",
            "AresW-03_C_cap_anc_local": "近祖先保护帽 C 端挂载方案。",
            "AresW-04_C_cap_anc_mid": "中祖先保护帽 C 端挂载方案。",
            "AresW-05_C_cap_anc_deep": "深祖先保护帽 C 端挂载方案。",
        },
    )
    cld_rows = [
        {
            "name": "CldNative_Q47CX0",
            "phase": "Cld 主线",
            "status": "天然参考骨架",
            "purpose": "天然参考骨架和结构映射锚点，后续所有祖先体与突变体都需要和它比较。",
            "length": "482 aa（含信号肽）",
            "notes": "作为 perchlorate/chlorite 场景下的主功能酶参考。",
        },
        {
            "name": "CldAnc_mid_Node007",
            "phase": "Cld ASR Round 1",
            "status": "优先表达候选",
            "purpose": "当前最像“稳中保活”的祖先骨架，是第一批表达和活性面板的优先对象。",
            "length": "成熟体工程长度级别",
            "notes": "适合与天然体及其他祖先体并排比较。",
        },
        {
            "name": "CldAnc_deep_Node012",
            "phase": "Cld ASR Round 1",
            "status": "深层祖先候选",
            "purpose": "更深层、更共识化的祖先骨架，主要用来赌稳定性与韧性上限。",
            "length": "成熟体工程长度级别",
            "notes": "更适合做“先赌稳定，再看活性”的路线。",
        },
        {
            "name": "CldAnc_localDistinct_Node006",
            "phase": "Cld ASR Round 1",
            "status": "近祖先对照候选",
            "purpose": "与 Node007 形成近祖先 AB 对照，用来观察局部共识变化的影响。",
            "length": "成熟体工程长度级别",
            "notes": "适合做表达、基础活性和氧化耐受差异对比。",
        },
    ]
    excipient_rows = [
        {
            "name": "CAHS family",
            "phase": "天然赋形剂面板",
            "status": "建议优先测试",
            "purpose": "作为冻干、脱水、低温保护的一线对照，验证蛋白赋形剂是否对 cargo 保活更有效。",
            "length": "类别面板",
            "notes": "主要对应脱水、冻干和冻融保护。",
        },
        {
            "name": "Group 3 LEA family",
            "phase": "天然赋形剂面板",
            "status": "建议优先测试",
            "purpose": "经典脱水保护类对照，用来比较与 CAHS 的差异。",
            "length": "类别面板",
            "notes": "主要对应干燥保护与低温保护。",
        },
        {
            "name": "高溶解度无序保护蛋白对照",
            "phase": "天然赋形剂面板",
            "status": "计划中",
            "purpose": "判断保护效应到底来自特殊机制还是一般的高亲水高溶解度性质。",
            "length": "类别面板",
            "notes": "偏机制对照组。",
        },
        {
            "name": "传统小分子赋形体系",
            "phase": "天然赋形剂面板",
            "status": "计划中",
            "purpose": "作为工业基线对照，比较蛋白赋形剂是否真的具有额外收益。",
            "length": "类别面板",
            "notes": "包括海藻糖等常见体系。",
        },
    ]
    return {
        "Cld 主功能酶路线": cld_rows,
        "Ares 首轮原型": ares_seed,
        "AresG Round 2 保护模块": ares_g2,
        "DrwH 祖先面板": drwh_anc,
        "AresW cargo-cap 拼接体": ares_w,
        "天然赋形剂与外部对照": excipient_rows,
    }


def build_priority_rows() -> list[dict[str, str]]:
    return [
        {
            "priority": "P0",
            "group": "Cld 主线表达与活性面板",
            "targets": "Q47CX0, Node007, Node012, Node006",
            "why": "这是当前主叙事最强的一组，既能讲 ASR 骨架，又能讲功能韧性与后续理性改造。",
            "next_action": "先做表达、heme 装配、基础 chlorite 活性，再做 H2O2、低温、干燥复水 readout。",
        },
        {
            "priority": "P1",
            "group": "Cld 结构映射后的保守改造",
            "targets": "Node007/Node012 上的口袋边缘与氧化热点小突变",
            "why": "这是把“祖先骨架”推进到“火星多压力功能韧性工程”的关键一步。",
            "next_action": "优先改 heme 口袋边缘和氧化热点，不要大动核心催化位点。",
        },
        {
            "priority": "P1",
            "group": "AresG2 保护模块验证",
            "targets": "AresG2-01, AresG2-02, AresG2-03",
            "why": "AresG 已经替代 AresR 成为主保护模块方向，现在最需要判定哪个长度组合最稳。",
            "next_action": "先做 AF3 与表达友好性判断，再进入冻干、低温、高渗和 cargo 保护 readout。",
        },
        {
            "priority": "P1",
            "group": "DrwH 祖先筛选",
            "targets": "DrwHAnc-01, DrwHAnc-02, DrwHAnc-03",
            "why": "DrwH 更适合做 cargo-cap，先确定哪种祖先深度最像稳定保护帽，再决定是否进入 AresW 拼接。",
            "next_action": "优先做 AF3、表达友好性和局部折叠稳定性比较。",
        },
        {
            "priority": "P2",
            "group": "AresW cargo-cap 拼接体验证",
            "targets": "AresW-03, AresW-04, AresW-05",
            "why": "这一步要建立“保护帽 + 底盘”的模块化能力，但前提是 DrwH 祖先层级已经收敛。",
            "next_action": "选择最稳的祖先 cap 后再推进，不建议现在平均推进全部构型。",
        },
        {
            "priority": "P2",
            "group": "天然赋形剂对照面板",
            "targets": "CAHS, Group 3 LEA, 小分子体系",
            "why": "这是主线实验的重要外部对照，决定我们是否真的需要新设计保护蛋白。",
            "next_action": "先做 CAHS/LEA 与小分子体系的最小对照矩阵。",
        },
    ]


def build_next_experiments() -> list[dict[str, str]]:
    return [
        {
            "rank": "1",
            "experiment": "Cld 第一版表达与功能韧性面板",
            "inputs": "Q47CX0, Node007, Node012, Node006",
            "goal": "确定天然体与祖先骨架在表达、heme 装配和基础活性上的可比性。",
            "readouts": "可溶表达, heme 装配, chlorite 基础活性, 4/10/20/30C 活性",
            "success": "至少有 1 到 2 条祖先骨架在表达或低温/应激活性上优于天然体。",
        },
        {
            "rank": "2",
            "experiment": "Cld 氧化与干燥复水挑战",
            "inputs": "Q47CX0, Node007, Node012, Node006",
            "goal": "把“火星相关多压力”从口头概念变成明确 readout。",
            "readouts": "H2O2 挑战后剩余活性, 干燥复水后剩余活性, 冻融后恢复",
            "success": "祖先骨架至少在一项 stress proxy 上明显优于天然体。",
        },
        {
            "rank": "3",
            "experiment": "AresG2 三版本 AF3 与表达优先级筛选",
            "inputs": "AresG2-01, AresG2-02, AresG2-03",
            "goal": "确定 short / balanced / long 三种构型里哪一版最适合继续推进。",
            "readouts": "AF3 core/tail 稳定性, 表达友好性, 可溶性趋势",
            "success": "收敛出 1 个主推版本和 1 个保留对照版本。",
        },
        {
            "rank": "4",
            "experiment": "DrwH 祖先域 AF3 与保护帽候选筛选",
            "inputs": "DrwHAnc-01, DrwHAnc-02, DrwHAnc-03",
            "goal": "判断近祖先、中祖先、深祖先谁更像紧凑稳定的 cargo-cap。",
            "readouts": "WHy 折叠紧凑性, 局部稳定性, 与 core 拼接前的独立可行性",
            "success": "确定 1 到 2 条最值得进入 AresW 的 cap 候选。",
        },
        {
            "rank": "5",
            "experiment": "天然赋形剂最小对照矩阵",
            "inputs": "CAHS, Group 3 LEA, 海藻糖等传统小分子体系",
            "goal": "判断保护效果是来自新设计模块还是常规赋形策略就已足够。",
            "readouts": "冻干恢复, 冻融恢复, 低温稳定, 干燥复水恢复",
            "success": "建立蛋白赋形剂与传统体系的基线差异。",
        },
    ]


def build_route_cards() -> list[dict[str, str]]:
    return [
        {
            "tag": "Cld 主线",
            "title": "先把主酶路线做扎实",
            "body": "当前最值得优先压实的不是再扩更多蛋白，而是先把 Cld 的天然体、祖先骨架和保守改造做出真正的 stress proxy 数据。",
            "meta": "优先 readout：chlorite 活性、H2O2、低温、干燥复水",
        },
        {
            "tag": "AresG",
            "title": "作为保护模块，不要抢主功能位",
            "body": "AresG 现在最合理的定位是保护微环境模块。它应该服务于 cargo 保活和应激保护，而不是先去背负全部火星生存叙事。",
            "meta": "优先 readout：AF3 平衡性、冻干/高渗保护、配方兼容性",
        },
        {
            "tag": "DrwH / AresW",
            "title": "先确定 cap 再谈拼接",
            "body": "DrwH 这条线的关键是先判断哪一层祖先最像稳定保护帽，再推进 AresW 拼接，而不是现在把所有拼接体一视同仁地推进。",
            "meta": "优先 readout：祖先域紧凑性、表达友好性、局部保护能力",
        },
        {
            "tag": "平台路线",
            "title": "方法平台跟着主线走",
            "body": "下一步的 benchmark 与 topic 配置更新，应该服务于 Cld / AresG / DrwH 主线，而不是为了扩 benchmark 而扩 benchmark。",
            "meta": "优先动作：把专题 readout 反向接入 MARS-FIELD 目标函数",
        },
    ]


def build_benchmark_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(MARS_ROOT / "outputs" / "benchmark_twelvepack_final" / "benchmark_summary.csv")
    purpose_map = {
        "1LBT": "CALB 脂肪酶模型，用来检验表面氧化热点识别和局部理性硬化策略。",
        "tem1_1btl": "标准 beta-lactamase scaffold，适合验证多位点工程组合的稳定性和可表达性。",
        "petase_5xfy": "PETase 结构状态之一，用来测试芳香位点硬化和 cutinase 家族迁移。",
        "petase_5xh3": "PETase 另一结构状态，用来验证相近 scaffold 下的策略一致性。",
        "sfgfp_2b3p": "报告蛋白，用来观察深度改造后信号输出和折叠是否保留。",
        "t4l_171l": "经典稳定性工程模型，测试 decoder 与选择策略的保守性。",
        "subtilisin_2st1": "工业蛋白酶 scaffold，验证跨 family 的普适性。",
        "adk_1s3g": "adenylate kinase，测试冷适应 family prior 的迁移作用。",
        "esterase_7b4q": "酯酶 family，测试 family prior 是否帮助环境适应设计。",
        "sod_1y67": "天然抗氧化主题相关目标，用于校准氧化压力下的设计逻辑。",
        "CLD_3Q09_NOTOPIC": "Cld 无专题约束对照组，用来衡量 topic-aware 设计的增益。",
        "CLD_3Q09_TOPIC": "Cld 带专题约束版本，是火星主线与平台路线对接的关键 target。",
    }
    positions = {
        "1LBT": "249, 251, 298",
        "tem1_1btl": "153, 155, 229, 272",
        "petase_5xfy": "3, 40, 41, 117",
        "petase_5xh3": "3, 40, 41, 117",
        "sfgfp_2b3p": "25, 139, 182, 231",
        "t4l_171l": "88, 126, 139, 158",
        "subtilisin_2st1": "21, 104, 241, 262",
        "adk_1s3g": "24, 28, 103, 109",
        "esterase_7b4q": "52, 61, 121, 123",
        "sod_1y67": "10, 18, 175, 180",
        "CLD_3Q09_NOTOPIC": "155, 156, 167, 212, 227",
        "CLD_3Q09_TOPIC": "155, 156, 167, 212, 227",
    }
    output: list[dict[str, str]] = []
    for row in rows:
        target = str(row["target"])
        output.append(
            {
                "target": target,
                "family": str(row["family"]),
                "design_positions": positions.get(target, ""),
                "role": purpose_map.get(target, ""),
                "overall": str(row["policy_mutations"] or row["overall_mutations"]),
                "overall_source": str(row["policy_source"] or row["overall_source"]),
                "engineering_score": str(row["policy_engineering_score"] or row["overall_mars_score"]),
                "decoder": "on" if str(row.get("decoder_enabled", "")).lower() == "true" else "off",
                "neural": "on" if str(row.get("neural_rerank_enabled", "")).lower() == "true" else "off",
            }
        )
    return output


def build_code_tables() -> dict[str, list[dict[str, str]]]:
    return {
        "仓库顶层布局": [
            {"path": "F:\\4-15Marsprotein\\scripts", "role": "火星专题脚本区", "desc": "Ares / Cld / DrwH / ASR 的专题脚本入口。"},
            {"path": "F:\\4-15Marsprotein\\designs", "role": "专题结果区", "desc": "AF3、ASR、结构映射、构建设计和变体面板结果。"},
            {"path": "F:\\4-15Marsprotein\\mars_stack", "role": "通用方法平台", "desc": "统一的 pipeline、benchmark、场构建、排序与图表生成。"},
            {"path": "F:\\4-15Marsprotein\\reports\\mars_resilience_stack", "role": "综合报告区", "desc": "三层框架总结、图表、实验记录与代码索引。"},
            {"path": "F:\\4-15Marsprotein\\release_packages", "role": "对外交付区", "desc": "HTML 交付、压缩包和后续正式版本落地位置。"},
        ],
        "核心算法模块": [
            {"file": "structure_features.py", "role": "结构特征提取", "desc": "SASA、B-factor、保护位点距离、氧化热点和柔性表面位点。"},
            {"file": "evolution.py", "role": "进化先验", "desc": "homolog profile、family prior、template-aware weighting。"},
            {"file": "ancestral_field.py", "role": "祖先后验场", "desc": "ASR posterior、entropy、confidence 和推荐位点。"},
            {"file": "retrieval_memory.py", "role": "结构检索记忆", "desc": "motif atlas、本地结构原型与 residue-level retrieval recommendation。"},
            {"file": "evidence_field.py", "role": "统一证据场构建", "desc": "把多源证据聚合成 position field。"},
            {"file": "energy_head.py", "role": "成对耦合能量", "desc": "构造 pairwise tensor，用于组合突变兼容性建模。"},
            {"file": "decoder.py", "role": "约束式解码器", "desc": "在 field + pairwise tensor 上做 beam decoding。"},
            {"file": "mars_score.py", "role": "通用工程打分", "desc": "氧化、表面、进化、负担等基础工程目标。"},
            {"file": "topic_score.py", "role": "专题目标层", "desc": "Cld / DrwH / AresG 的定向加权逻辑。"},
            {"file": "fusion_ranker.py", "role": "学习式融合排序", "desc": "对候选做 feature-level 排序、校准与 hybrid policy 选择。"},
        ],
        "field_network 抽象层": [
            {"file": "contracts.py", "role": "证据契约", "desc": "定义 context、evidence bundle 和 residue field 的标准接口。"},
            {"file": "encoders.py", "role": "编码器抽象", "desc": "结构、进化、祖先、检索和环境分支的统一抽象层。"},
            {"file": "residue_field.py", "role": "统一 residue field", "desc": "将多分支证据投影到共享决策空间。"},
            {"file": "neural_dataset.py", "role": "神经训练数据", "desc": "把 target runtime 状态转成神经分支可训练的数据对象。"},
            {"file": "neural_model.py", "role": "神经场模型", "desc": "学习 unary、pairwise、gates 和 candidate-level signals。"},
            {"file": "neural_generator.py", "role": "神经解码生成", "desc": "生成 neural residue field 与 neural pairwise tensor。"},
            {"file": "neural_training.py", "role": "训练与 holdout", "desc": "负责 holdout 训练和 pairwise consistency 学习。"},
            {"file": "system.py", "role": "统一调度系统", "desc": "把 build_evidence、construct_field、decode 串起来。"},
            {"file": "scoring.py", "role": "候选行打分", "desc": "将 candidate rows 接入 MarsScore 与 topic score。"},
            {"file": "proposals.py", "role": "候选生成基础操作", "desc": "维护 source priority、local proposals 和候选注册逻辑。"},
        ],
        "主要执行入口": [
            {"file": "run_mars_pipeline.py", "role": "单目标主流程", "desc": "生成候选、构场、排序、decoder、neural rerank 和输出。"},
            {"file": "run_mars_benchmark.py", "role": "多目标 benchmark", "desc": "统一跑 twelvepack / ninepack，并输出 family/held-out/ablation。"},
            {"file": "build_structure_motif_atlas.py", "role": "motif atlas 构建", "desc": "建立 retrieval branch 的本地结构模体记忆。"},
            {"file": "run_mars_field_neural_reranker.py", "role": "神经重排入口", "desc": "针对单 target 训练 holdout neural reranker。"},
            {"file": "render_mars_field_architecture_v2.py", "role": "架构图生成", "desc": "渲染 MARS-FIELD 总体架构图。"},
            {"file": "render_mars_field_benchmark_overview_v3.py", "role": "benchmark 图", "desc": "渲染 twelvepack 总览图。"},
            {"file": "render_mars_field_decoder_analysis_v3.py", "role": "decoder 分析图", "desc": "渲染 decoder 和 calibration 结果。"},
            {"file": "render_mars_field_mechanism_limitations_v5.py", "role": "机制/边界图", "desc": "呈现 gains、limitations 和机制解释。"},
            {"file": "render_case_study_structure_assets.py", "role": "结构案例资产", "desc": "输出 PyMOL 结构窗口、overview 和会话文件。"},
            {"file": "build_mars_field_paper_bundle.py", "role": "论文资产打包", "desc": "打包 figure、table、case-study 和摘要文稿。"},
        ],
    }


def build_tech_route_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": "1. 输入定义",
            "what": "读取 target 配置、结构、序列、设计位点、保护位点和同源/祖先数据路径。",
            "files": "configs/*.yaml, run_mars_pipeline.py",
            "output": "统一的 ProteinDesignContext 与 EvidencePaths。",
        },
        {
            "stage": "2. 多源证据提取",
            "what": "提取结构特征、进化 profile、祖先 posterior、retrieval motif 和记忆、环境条件代理。",
            "files": "structure_features.py, evolution.py, ancestral_field.py, retrieval_memory.py",
            "output": "Geometric / Evolution / Ancestral / Retrieval / Environment evidence。",
        },
        {
            "stage": "3. 共享 residue field 构建",
            "what": "把多源证据写入共享 residue decision space，构造 position field 与 pairwise tensor。",
            "files": "evidence_field.py, energy_head.py, field_network/system.py",
            "output": "可解码的 residue field 和 pairwise coupling。",
        },
        {
            "stage": "4. 候选生成",
            "what": "并行保留人工理性候选、ProteinMPNN、ESM-IF、local proposals、fusion_decoder、neural_decoder。",
            "files": "run_mars_pipeline.py, field_network/proposals.py, decoder.py",
            "output": "多路 proposal 候选池。",
        },
        {
            "stage": "5. 排序与混合策略选择",
            "what": "使用 MarsScore、topic-aware score、learned fusion ranker、neural reranker 和 hybrid policy 做重排。",
            "files": "mars_score.py, topic_score.py, fusion_ranker.py, run_mars_field_neural_reranker.py",
            "output": "overall winner、best learned、policy winner 和神经比较结果。",
        },
        {
            "stage": "6. 资产输出与汇报",
            "what": "输出 csv / json / fasta / summary、benchmark、图表、结构案例和 paper bundle。",
            "files": "run_mars_benchmark.py, render_*.py, build_mars_field_paper_bundle.py",
            "output": "可复查、可汇报、可写文章的完整研究资产。",
        },
    ]


def build_figure_assets() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    topic_figures = [
        {
            "label": "Topic Figure 1",
            "title": "三层统一框架",
            "caption": "把项目收敛成三层：Cld 主功能层、AresG/DrwH 保护模块层、MarsStack 方法平台层。",
            "src": copy_file(
                ROOT / "reports" / "mars_resilience_stack" / "figures" / "figure1_three_layer_framework.png",
                IMG_ROOT / "topic" / "figure1_three_layer_framework.png",
            ),
        },
        {
            "label": "Topic Figure 2",
            "title": "Ares Round 1 AF3 汇总",
            "caption": "首轮 AF3 比较表明 AresG 比 AresR 更值得继续推进，主线从 DNA-tail 转向保护微环境。",
            "src": copy_file(
                ROOT / "reports" / "mars_resilience_stack" / "figures" / "figure2_ares_round1_af3.png",
                IMG_ROOT / "topic" / "figure2_ares_round1_af3.png",
            ),
        },
        {
            "label": "Topic Figure 3",
            "title": "ASR 候选深度分层",
            "caption": "展示 Cld / DrwH 路线中近祖先、中祖先、深祖先的候选层次，帮助决定表达和结构筛选顺序。",
            "src": copy_file(
                ROOT / "reports" / "mars_resilience_stack" / "figures" / "figure3_asr_candidate_depth.png",
                IMG_ROOT / "topic" / "figure3_asr_candidate_depth.png",
            ),
        },
        {
            "label": "Topic Figure 4",
            "title": "MarsStack benchmark 概览",
            "caption": "展示平台路线已经具备跨蛋白 benchmark 能力，不再停留在单一案例脚本阶段。",
            "src": copy_file(
                ROOT / "reports" / "mars_resilience_stack" / "figures" / "figure4_marsstack_benchmark.png",
                IMG_ROOT / "topic" / "figure4_marsstack_benchmark.png",
            ),
        },
        {
            "label": "Topic Figure 5",
            "title": "Mars objective 消融",
            "caption": "说明 oxidation / surface / evolution 等目标项不是装饰性规则，而会真实改变 top candidate 排序。",
            "src": copy_file(
                ROOT / "reports" / "mars_resilience_stack" / "figures" / "figure5_mars_objective_ablation.png",
                IMG_ROOT / "topic" / "figure5_mars_objective_ablation.png",
            ),
        },
        {
            "label": "Topic Figure 6",
            "title": "AresG Round 2 与变体面板",
            "caption": "总结 AresG2 的 short / balanced / long 三个版本，是后续 AF3 和保护实验的重点输入。",
            "src": copy_file(
                ROOT / "reports" / "mars_resilience_stack" / "figures" / "figure6_round2_and_variant_mix.png",
                IMG_ROOT / "topic" / "figure6_round2_and_variant_mix.png",
            ),
        },
    ]
    paper_figures = [
        {
            "label": "Figure 1",
            "title": "MARS-FIELD 架构图",
            "caption": "展示几何、进化、祖先、retrieval 和记忆、环境分支如何进入共享 residue field 和 pairwise tensor。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure1_mars_field_architecture_v2.png",
                IMG_ROOT / "paper" / "figure1_mars_field_architecture_v2.png",
            ),
        },
        {
            "label": "Figure 2",
            "title": "Twelvepack benchmark 总览",
            "caption": "呈现 12 个 target、10 个 family 的平台级验证结果，是方法论证的主图之一。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure2_benchmark_overview_v3.png",
                IMG_ROOT / "paper" / "figure2_benchmark_overview_v3.png",
            ),
        },
        {
            "label": "Figure 3",
            "title": "Decoder 与 calibration 分析",
            "caption": "展示 decoder 引入、candidate 校准与 final policy 选择之间的关系。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure3_decoder_calibration_v3.png",
                IMG_ROOT / "paper" / "figure3_decoder_calibration_v3.png",
            ),
        },
        {
            "label": "Figure 3B",
            "title": "机制与限制分析",
            "caption": "用于诚实描述 gains 与 limitations，避免把当前版本写成 fully joint 的终版模型。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure3_mechanism_limitations_v5.png",
                IMG_ROOT / "paper" / "figure3_mechanism_limitations_v5.png",
            ),
        },
        {
            "label": "Figure 3C",
            "title": "Neural comparison",
            "caption": "比较当前 overall policy 与 neural branch 在不同 target 上的关系与增益。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure_neural_comparison_v1.png",
                IMG_ROOT / "paper" / "figure_neural_comparison_v1.png",
            ),
        },
        {
            "label": "Figure 3D",
            "title": "Neural branch diagnostics",
            "caption": "展示神经分支的 gate 行为和诊断信息，用于解释哪类 target 更受神经分支影响。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure_neural_branch_diagnostics_v1.png",
                IMG_ROOT / "paper" / "figure_neural_branch_diagnostics_v1.png",
            ),
        },
        {
            "label": "Figure 3E",
            "title": "Policy comparison",
            "caption": "展示 current / neural / hybrid 三种 policy 的区别，说明为何当前默认仍是 hybrid。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure_policy_compare_v1.png",
                IMG_ROOT / "paper" / "figure_policy_compare_v1.png",
            ),
        },
        {
            "label": "Figure 4",
            "title": "Case studies master",
            "caption": "把 1LBT、TEM1、PETase、Cld 四个代表案例收敛成一个总面板，用于论文或汇报概览。",
            "src": copy_file(
                MARS_ROOT / "outputs" / "paper_bundle_v1" / "figures" / "figure4_case_studies_master_v3.png",
                IMG_ROOT / "paper" / "figure4_case_studies_master_v3.png",
            ),
        },
    ]
    structure_manifest = read_json(MARS_ROOT / "outputs" / "paper_bundle_v1" / "structure_panels" / "structure_panel_manifest.json")
    target_notes = {
        "1LBT": "紧凑 benchmark 案例，适合展示单点/小窗口设计如何被平台重写方向。",
        "tem1_1btl": "多位点工程案例，展示 selector 稳定性与工程一致性。",
        "petase_5xh3": "PETase 家族迁移案例，展示相近 scaffold 下的设计一致性。",
        "CLD_3Q09_TOPIC": "火星主线锚点案例，展示 topic-aware scoring 对 Cld 的增益。",
        "CLD_3Q09_NOTOPIC": "与 topic 版配对，用来展示无专题约束时的差异。",
    }
    structure_cards: list[dict[str, str]] = []
    for entry in structure_manifest["targets"]:
        target = str(entry["target"])
        if target not in target_notes:
            continue
        structure_cards.append(
            {
                "label": target,
                "title": f"{target} overview",
                "caption": target_notes[target],
                "src": copy_file(Path(entry["overview_png"]), IMG_ROOT / "structure" / f"{target}_overview.png"),
            }
        )
        structure_cards.append(
            {
                "label": f"{target} window",
                "title": f"{target} design window",
                "caption": "局部设计窗口渲染，用来观察关键位点和局部空间环境的关系。",
                "src": copy_file(Path(entry["design_window_png"]), IMG_ROOT / "structure" / f"{target}_design_window.png"),
            }
        )
    return topic_figures, paper_figures, structure_cards


def build_downloads() -> list[dict[str, str]]:
    files = [
        ("Word", "Mars-4-19.docx", ROOT / "Mars-4-19.docx", "用户提供的参考文稿。"),
        ("Markdown", "中文宏观报告草稿 v2", MARS_ROOT / "docs" / "mars_field_macro_report_cn_draft_v2.md", "当前中文版总报告草稿。"),
        ("CSV", "benchmark_summary.csv", MARS_ROOT / "outputs" / "benchmark_twelvepack_final" / "benchmark_summary.csv", "十二目标 benchmark 汇总表。"),
        ("Markdown", "benchmark_summary.md", MARS_ROOT / "outputs" / "benchmark_twelvepack_final" / "benchmark_summary.md", "十二目标 benchmark 的文字版摘要。"),
        ("Markdown", "paper_bundle_summary.md", MARS_ROOT / "outputs" / "paper_bundle_v1" / "bundle_summary.md", "当前 paper bundle 的内容清单与 case study 说明。"),
        ("Markdown", "mars_resilience_stack_report.md", ROOT / "reports" / "mars_resilience_stack" / "mars_resilience_stack_report.md", "三层统一框架报告。"),
        ("CSV", "experiment_log.csv", ROOT / "reports" / "mars_resilience_stack" / "experiment_log.csv", "专题路线和平台路线的实验/计算记录索引。"),
        ("CSV", "code_asset_inventory.csv", ROOT / "reports" / "mars_resilience_stack" / "code_asset_inventory.csv", "当前工作区代码与资产归档清单。"),
        ("Zip", "mars_field_html_delivery_v2_2_final.zip", ZIP_PATH, "本次 HTML 包的压缩版，生成结束后会自动补入。"),
    ]
    output: list[dict[str, str]] = []
    for file_type, name, src, note in files:
        if not src.exists():
            continue
        href = copy_file(src, DOWNLOAD_ROOT / src.name)
        output.append({"type": file_type, "name": name, "href": href, "note": note})
    return output


def build_runbook_commands() -> list[dict[str, str]]:
    return [
        {
            "tag": "Environment",
            "title": "进入平台根目录",
            "command": "cd F:\\4-15Marsprotein\\mars_stack",
            "desc": "MARS-FIELD 平台相关命令建议都在 mars_stack 根目录下执行。",
            "meta": "后续所有 configs / scripts / outputs 都默认相对这个目录解析。",
        },
        {
            "tag": "Sanity Check",
            "title": "检查数据布局",
            "command": "python scripts\\validate_dataset_layout.py",
            "desc": "用于检查 inputs、datasets、configs 和 outputs 所需的基本目录结构是否可用。",
            "meta": "适合在换机器或重新打包后先运行一次。",
        },
        {
            "tag": "Atlas",
            "title": "构建结构 motif atlas",
            "command": "python scripts\\build_structure_motif_atlas.py",
            "desc": "为 retrieval branch 建立本地结构模体记忆库。",
            "meta": "输出会进入 outputs 下的 atlas 相关文件。",
        },
        {
            "tag": "Single Target",
            "title": "运行单目标 pipeline（Cld topic 版示例）",
            "command": "python scripts\\run_mars_pipeline.py --config configs\\cld_3q09_topic.yaml --top-k 12",
            "desc": "跑一个完整 target 的候选生成、field 构建、排序、decoder 和输出流程。",
            "meta": "输出目录示例：outputs\\cld_3q09_topic_pipeline",
        },
        {
            "tag": "Single Target",
            "title": "运行单目标 pipeline（TEM-1 示例）",
            "command": "python scripts\\run_mars_pipeline.py --config configs\\tem1_1btl.yaml --top-k 12",
            "desc": "这是 README 里的典型单目标示例，适合快速验证系统是否正常。",
            "meta": "输出目录示例：outputs\\tem1_1btl_pipeline",
        },
        {
            "tag": "Benchmark",
            "title": "运行 twelvepack final benchmark",
            "command": "python scripts\\run_mars_benchmark.py --benchmark-config configs\\benchmark_twelvepack_final.yaml --top-k 12",
            "desc": "统一跑 12 个 target，并输出 benchmark summary、family、held-out 和 ablation 结果。",
            "meta": "输出目录：outputs\\benchmark_twelvepack_final",
        },
        {
            "tag": "Neural",
            "title": "单 target neural reranker",
            "command": "python scripts\\run_mars_field_neural_reranker.py --target 1LBT --epochs 1 --lr 0.001",
            "desc": "训练/运行 holdout neural reranker，用于比较 current 与 neural branch 行为。",
            "meta": "可把 target 替换成 1LBT, tem1_1btl, CLD_3Q09_TOPIC 等。",
        },
        {
            "tag": "Figures",
            "title": "渲染主图与数据图",
            "command": "python scripts\\render_mars_field_architecture_v2.py\npython scripts\\render_mars_field_benchmark_overview_v3.py\npython scripts\\render_mars_field_decoder_analysis_v3.py\npython scripts\\render_mars_field_mechanism_limitations_v5.py",
            "desc": "更新架构图、benchmark 总图、decoder 分析和机制/边界图。",
            "meta": "输出目录：outputs\\paper_bundle_v1\\figures",
        },
        {
            "tag": "Bundle",
            "title": "生成 paper bundle",
            "command": "python scripts\\build_mars_field_paper_bundle.py",
            "desc": "把表格、figure、case study 和文稿资产打包成当前论文资产目录。",
            "meta": "输出目录：outputs\\paper_bundle_v1",
        },
        {
            "tag": "Structure",
            "title": "生成结构案例资产",
            "command": "python scripts\\render_case_study_structure_assets.py --render --include-companions\npython scripts\\render_case_study_composites_v2.py",
            "desc": "生成 PyMOL 场景、overview 图、design window 和 case-study composites。",
            "meta": "输出目录：outputs\\paper_bundle_v1\\structure_panels",
        },
        {
            "tag": "Topical Script",
            "title": "运行 Cld ASR 第一轮",
            "command": "python scripts\\run_cld_asr.py --input-fasta designs\\asr_cld_prb\\cld_asr_input.fasta --center Q47CX0 --output-dir designs\\asr_cld_prb\\run3 --skip-iqtree",
            "desc": "快速重现 Cld 第一轮工程版 ASR 结果。",
            "meta": "适合在 Windows 环境先做可执行第一轮，再准备更严格版。",
        },
        {
            "tag": "Topical Script",
            "title": "运行 DrwH ASR 第一轮",
            "command": "python scripts\\run_drwh_asr.py --input-fasta designs\\asr_drwh\\asr_input_whydomain.fasta --center Q9RUL2 --output-dir designs\\asr_drwh\\run4 --skip-iqtree",
            "desc": "快速重现 DrwH 祖先面板的第一轮结果。",
            "meta": "适合先比较近祖先、中祖先、深祖先的结构趋势。",
        },
        {
            "tag": "Topical Script",
            "title": "为 Ares / AresG / AresW FASTA 打分",
            "command": "python scripts\\score_ares_fasta.py designs\\aresg_round2_constructs.fasta",
            "desc": "把构建体 FASTA 转成基础理化指标表，是 Ares 路线常用的快速打分入口。",
            "meta": "输出会写到同名 .scores.csv 文件。",
        },
        {
            "tag": "Reporting",
            "title": "生成三层框架综合报告",
            "command": "python scripts\\build_mars_resilience_stack_report.py",
            "desc": "整理 Cld / AresG / DrwH / MarsStack 的综合图表、代码索引与实验记录。",
            "meta": "输出目录：reports\\mars_resilience_stack",
        },
        {
            "tag": "Delivery",
            "title": "重新生成本 HTML 交付包",
            "command": "python mars_stack\\scripts\\build_mars_field_html_delivery.py",
            "desc": "重建 final HTML 交付目录与 zip 包。",
            "meta": "输出目录：release_packages\\mars_field_html_delivery_v2_2_final",
        },
    ]


def build_workflows() -> list[dict[str, str]]:
    return [
        {
            "tag": "Workflow A",
            "title": "主线实验准备流程",
            "body": "先锁定 Cld 天然体与 3 条祖先骨架，再并行准备氧化挑战、低温、干燥复水 readout。不要在主线骨架还没收敛前把大量保护模块实验一起推。",
            "meta": "顺序：Cld expression -> baseline activity -> stress proxy -> selective mutation round",
        },
        {
            "tag": "Workflow B",
            "title": "保护模块推进流程",
            "body": "AresG 和 DrwH 先做 AF3 / 表达友好性收敛，再进入 cargo 保护或配方测试。先决条件是主线功能酶已有明确对照 readout。",
            "meta": "顺序：AF3 -> expression-friendliness -> choose 1-2 winners -> cargo protection assay",
        },
        {
            "tag": "Workflow C",
            "title": "平台结果刷新流程",
            "body": "更新 retrieval atlas、单 target pipeline、十二目标 benchmark、图表和 paper bundle。只有在 topic scoring 或 selector 行为发生明显变化时再全量重跑。",
            "meta": "顺序：atlas -> single target -> benchmark -> figures -> paper bundle",
        },
    ]


def build_download_specs() -> list[tuple[str, str, Path, str]]:
    return [
        ("Word", "Mars-4-19.docx", ROOT / "Mars-4-19.docx", "用户提供的参考文稿。"),
        ("Markdown", "中文宏观报告草稿 v2", MARS_ROOT / "docs" / "mars_field_macro_report_cn_draft_v2.md", "当前中文版总报告草稿。"),
        ("CSV", "benchmark_summary.csv", MARS_ROOT / "outputs" / "benchmark_twelvepack_final" / "benchmark_summary.csv", "十二目标 benchmark 汇总表。"),
        ("Markdown", "benchmark_summary.md", MARS_ROOT / "outputs" / "benchmark_twelvepack_final" / "benchmark_summary.md", "十二目标 benchmark 的文字版摘要。"),
        ("Markdown", "paper_bundle_summary.md", MARS_ROOT / "outputs" / "paper_bundle_v1" / "bundle_summary.md", "当前 paper bundle 的内容清单与 case study 说明。"),
        ("Markdown", "mars_resilience_stack_report.md", ROOT / "reports" / "mars_resilience_stack" / "mars_resilience_stack_report.md", "三层统一框架报告。"),
        ("CSV", "experiment_log.csv", ROOT / "reports" / "mars_resilience_stack" / "experiment_log.csv", "专题路线和平台路线的实验/计算记录索引。"),
        ("CSV", "code_asset_inventory.csv", ROOT / "reports" / "mars_resilience_stack" / "code_asset_inventory.csv", "当前工作区代码与资产归档清单。"),
    ]


def write_downloads(include_zip: bool = False) -> list[dict[str, str]]:
    specs = build_download_specs()
    if include_zip:
        specs.append(("Zip", "mars_field_html_delivery_v2_2_final.zip", ZIP_PATH, "本次 HTML 包的压缩版。"))
    output: list[dict[str, str]] = []
    for file_type, name, src, note in specs:
        if not src.exists():
            continue
        href = copy_file(src, DOWNLOAD_ROOT / src.name)
        output.append({"type": file_type, "name": name, "href": href, "note": note})
    return output


def page_nav(active: str) -> str:
    items = [
        ("index.html", "总览", "overview"),
        ("experiments.html", "实验优先级", "experiments"),
        ("runbook.html", "运行手册", "runbook"),
    ]
    links = []
    for href, label, key in items:
        cls = "active" if key == active else ""
        links.append(f'<a class="{cls}" href="{href}">{html_escape(label)}</a>')
    return "\n".join(links)


def page_shell(title: str, subtitle: str, active: str, stats_html: str, body_html: str) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_escape(title)}</title>
  <link rel="stylesheet" href="assets/style.css" />
</head>
<body>
  <header class="hero" id="top">
    <div class="hero-inner">
      <div class="eyebrow">MARS-FIELD / MarsStack / HTML Delivery Final</div>
      <h1>{html_escape(title)}</h1>
      <p class="hero-lead">{html_escape(subtitle)}</p>
      <div class="hero-meta">生成时间：{html_escape(generated_at)} | 目录：{html_escape(str(OUT_ROOT))}</div>
      <div class="stats-grid">{stats_html}</div>
    </div>
  </header>
  <nav class="page-nav">{page_nav(active)}</nav>
  <main class="container">{body_html}</main>
  <footer class="page-footer">
    <div>HTML 交付包 Final | Generated by <code>mars_stack/scripts/build_mars_field_html_delivery.py</code></div>
    <a href="#top">回到顶部</a>
  </footer>
</body>
</html>
"""


def build_index_html() -> str:
    stats = [
        {"value": "2.2 Final", "label": "交付版本", "note": "多页面 HTML + zip"},
        {"value": "12", "label": "Benchmark targets", "note": "当前 twelvepack final 面板"},
        {"value": "10", "label": "Families", "note": "平台级跨 family 验证"},
        {"value": "3", "label": "主线专题", "note": "Cld / AresG / DrwH"},
        {"value": "4", "label": "主案例", "note": "1LBT / TEM1 / PETase / Cld"},
        {"value": "1", "label": "统一平台", "note": "MARS-FIELD / MarsStack"},
    ]
    summary_cards = [
        {
            "tag": "主功能层",
            "title": "Cld 主线",
            "body": "围绕 perchlorate/chlorite 相关解毒功能，推进 ASR、结构映射、抗氧化理性改造和后续活性面板，是当前最适合收敛为主叙事的路线。",
            "meta": "关键词：Q47CX0 / Node007 / Node012 / Node006",
        },
        {
            "tag": "保护模块层",
            "title": "AresG / DrwH",
            "body": "AresG 更偏向应激下形成可逆保护微环境，DrwH 更适合作为 cargo-cap 或局部保护帽。它们是主线的保护增强模块，不是主功能酶本身。",
            "meta": "关键词：AresG2 / DrwH ancestors / AresW cargo-cap",
        },
        {
            "tag": "方法平台层",
            "title": "MARS-FIELD / MarsStack",
            "body": "把结构、进化、祖先、retrieval 和记忆、环境条件和候选解码纳入统一 residue field，再通过排序和 hybrid policy 输出稳定可解释的设计资产。",
            "meta": "关键词：evidence-to-sequence / residue field / pairwise tensor",
        },
    ]
    repo_tree = """F:\\4-15Marsprotein
├─ scripts/                      # 火星专题脚本
├─ designs/                      # AF3 / ASR / 变体设计结果
├─ mars_stack/
│  ├─ configs/                   # target 与 benchmark 配置
│  ├─ docs/                      # README、路线、报告草稿
│  ├─ marsstack/                 # 核心算法库
│  ├─ scripts/                   # pipeline / benchmark / render 入口
│  └─ outputs/                   # benchmark、图表、case study 资产
├─ reports/mars_resilience_stack/# 三层框架综合报告
└─ release_packages/             # 对外交付目录
"""
    mainline_tables = build_mainline_tables()
    topic_figures, paper_figures, structure_cards = build_figure_assets()
    benchmark_rows = build_benchmark_rows()
    code_tables = build_code_tables()
    tech_rows = build_tech_route_rows()
    downloads = write_downloads(include_zip=True)

    mainline_sections = []
    mainline_columns = [
        ("name", "蛋白 / 构建体"),
        ("phase", "阶段"),
        ("status", "当前状态"),
        ("purpose", "作用 / 为什么做"),
        ("length", "长度或类别"),
        ("notes", "补充说明"),
    ]
    for title, rows in mainline_tables.items():
        mainline_sections.append(
            f"<section class='subsection'><h3>{html_escape(title)}</h3>{render_table(mainline_columns, rows)}</section>"
        )

    benchmark_columns = [
        ("target", "Target"),
        ("family", "Family"),
        ("design_positions", "设计位点"),
        ("role", "为什么选它"),
        ("overall", "当前 policy winner"),
        ("overall_source", "来源"),
        ("engineering_score", "engineering score"),
        ("decoder", "decoder"),
        ("neural", "neural"),
    ]
    code_column_map = {
        "仓库顶层布局": [("path", "路径"), ("role", "角色"), ("desc", "说明")],
        "核心算法模块": [("file", "文件"), ("role", "角色"), ("desc", "说明")],
        "field_network 抽象层": [("file", "文件"), ("role", "角色"), ("desc", "说明")],
        "主要执行入口": [("file", "文件"), ("role", "角色"), ("desc", "说明")],
    }
    code_sections = []
    for title, rows in code_tables.items():
        code_sections.append(f"<section class='subsection'><h3>{html_escape(title)}</h3>{render_table(code_column_map[title], rows)}</section>")

    tech_columns = [
        ("stage", "阶段"),
        ("what", "做什么"),
        ("files", "核心代码"),
        ("output", "输出对象"),
    ]
    body = f"""
    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Overview</div>
        <h2>项目总览</h2>
      </div>
      <p>
        当前项目最准确的理解方式，不是“一个火星蛋白项目”或者“一个单一模型项目”，而是一个三层并行推进的系统。
        第一层是 <strong>Cld 主功能酶路线</strong>，第二层是 <strong>AresG / DrwH 保护模块路线</strong>，
        第三层是 <strong>MARS-FIELD / MarsStack 方法平台</strong>。三者共同组成当前工作的完整闭环。
      </p>
      <div class="card-grid">{render_cards(summary_cards)}</div>
      <div class="note-box">
        <strong>当前版本边界：</strong>
        它已经是一个统一证据场驱动的研究原型平台，但还不是 fully joint、fully learned 的终版模型。
      </div>
      <div class="subsection">
        <h3>仓库树形视图</h3>
        <pre class="tree-block">{html_escape(repo_tree)}</pre>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Protein List</div>
        <h2>实验与改造蛋白清单</h2>
      </div>
      <p>这里把当前工作对象拆成两类：主线专题蛋白与构建体，以及证明平台通用性的 twelvepack benchmark 蛋白。</p>
      {''.join(mainline_sections)}
      <section class="subsection">
        <h3>平台 benchmark 蛋白（twelvepack final）</h3>
        <p class="section-note">这 12 个 target 主要是方法论证资产，不等于最终的生物学主线。</p>
        {render_table(benchmark_columns, benchmark_rows)}
      </section>
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Pipeline</div>
        <h2>技术路线与执行细节</h2>
      </div>
      <p>
        当前系统的主流程可以概括为六步：输入定义、多源证据提取、共享 residue field 构建、多路 proposal 候选生成、
        排序与 hybrid policy 选择、以及图表与数据资产输出。
      </p>
      {render_table(tech_columns, tech_rows)}
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Code Map</div>
        <h2>代码 README 布局与模块说明</h2>
      </div>
      <p>下面的模块说明按“读仓库最合理的顺序”组织，方便直接拿去做 README、交接或汇报。</p>
      {''.join(code_sections)}
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Figures</div>
        <h2>关键结果图</h2>
      </div>
      <p>
        为避免图和图例错位，这里统一采用固定卡片布局：图片单独居中，图题和说明在图片下方独立显示。
      </p>
      <section class="subsection">
        <h3>专题路线与三层框架图</h3>
        <div class="figure-grid">{render_figure_cards(topic_figures)}</div>
      </section>
      <section class="subsection">
        <h3>MARS-FIELD 平台与 paper bundle 图</h3>
        <div class="figure-grid">{render_figure_cards(paper_figures)}</div>
      </section>
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Structures</div>
        <h2>结构案例与设计窗口</h2>
      </div>
      <p>这里收录代表性 case-study 的 overview 和 design window 图，用来快速定位关键突变窗口与局部空间环境。</p>
      <div class="figure-grid">{render_figure_cards(structure_cards, "compact")}</div>
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Downloads</div>
        <h2>下载区</h2>
      </div>
      <p>收录了当前最常用的文稿、benchmark 表、paper bundle 摘要和专题路线记录，便于离线阅读和继续整理成 Word / PPT。</p>
      <div class="download-grid">{render_downloads(downloads)}</div>
      <div class="note-box">
        下一页 <a href="experiments.html">实验优先级</a> 会把主线蛋白重新按优先级排序，并给出现在最该先做的 5 组实验。
      </div>
    </section>
    """
    return page_shell(
        "MARS-FIELD HTML 交付包 Final",
        "把项目总览、蛋白清单、技术路线、代码布局和关键图表整理进一个可直接交付的 HTML 包。",
        "overview",
        render_stats(stats),
        body,
    )


def build_experiments_html() -> str:
    stats = [
        {"value": "P0-P2", "label": "优先级层级", "note": "主线优先于扩线"},
        {"value": "5", "label": "近期实验", "note": "现在最值得先做的五组实验"},
        {"value": "1", "label": "主叙事锚点", "note": "Cld 主线优先压实"},
        {"value": "2", "label": "保护模块方向", "note": "AresG 与 DrwH 并行筛选"},
    ]
    priority_columns = [
        ("priority", "优先级"),
        ("group", "对象组"),
        ("targets", "蛋白 / 构建体"),
        ("why", "为什么现在做"),
        ("next_action", "下一动作"),
    ]
    experiment_columns = [
        ("rank", "序号"),
        ("experiment", "实验"),
        ("inputs", "输入对象"),
        ("goal", "目标"),
        ("readouts", "关键 readout"),
        ("success", "成功判据"),
    ]
    body = f"""
    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Priority</div>
        <h2>主线蛋白优先级重排</h2>
      </div>
      <p>
        这一页的原则只有一个：不要再把所有对象平均推进。当前最强主叙事是 <strong>Cld 在火星相关多压力条件下的功能韧性工程路线</strong>，
        所以优先级必须先围绕 Cld 压实，再让 AresG / DrwH 作为保护增强模块逐步接上去。
      </p>
      {render_table(priority_columns, build_priority_rows())}
      <div class="card-grid">{render_cards(build_route_cards())}</div>
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Next Experiments</div>
        <h2>我们现在最该先做哪几个实验</h2>
      </div>
      <p>
        下面这 5 组实验按“最能决定项目方向”的顺序排列。它们不是平均分布的任务清单，而是最值得优先投入资源的验证链。
      </p>
      {render_table(experiment_columns, build_next_experiments())}
      <div class="note-box">
        <strong>当前最不建议的做法：</strong>
        在 Cld 骨架和 stress proxy readout 还没收敛前，就把 AresG、DrwH、天然赋形剂、更多 benchmark 和更多新蛋白全部一起推进。
        那样会让叙事和实验资源同时失焦。
      </div>
    </section>
    """
    return page_shell(
        "MARS-FIELD 实验优先级与近期实验页",
        "把主线蛋白按优先级重排，并明确现在最值得先做的实验顺序。",
        "experiments",
        render_stats(stats),
        body,
    )


def build_runbook_html() -> str:
    stats = [
        {"value": "14", "label": "常用命令", "note": "平台与专题路线主入口"},
        {"value": "3", "label": "推荐工作流", "note": "实验、保护模块、平台结果刷新"},
        {"value": "1", "label": "统一根目录", "note": "mars_stack 为平台默认执行目录"},
        {"value": "2", "label": "路线类型", "note": "专题脚本 + 平台脚本"},
    ]
    workflows = build_workflows()
    body = f"""
    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Runbook</div>
        <h2>代码入口怎么跑</h2>
      </div>
      <p>
        这一页把当前最常用的代码入口整理成可直接复制的运行手册。平台相关命令建议在 <code>F:\\4-15Marsprotein\\mars_stack</code> 下执行；
        火星专题路线脚本则在工作区根目录 <code>F:\\4-15Marsprotein</code> 下执行。
      </p>
      <div class="command-grid">{render_command_cards(build_runbook_commands())}</div>
    </section>

    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Workflow</div>
        <h2>推荐操作流</h2>
      </div>
      <p>如果不是一次性重跑全部结果，建议按下面 3 条工作流使用当前代码库。</p>
      <div class="card-grid">{render_cards(workflows)}</div>
      <div class="note-box">
        <strong>经验建议：</strong>
        只有在 topic scoring、selector 行为或 retrieval atlas 发生明显变化时，才建议全量重跑 twelvepack 和 paper bundle。否则先跑单 target 检查变化是否值得扩散。
      </div>
    </section>
    """
    return page_shell(
        "MARS-FIELD 代码运行手册",
        "把当前平台主入口、专题脚本和推荐操作流整理成可直接复制的 runbook。",
        "runbook",
        render_stats(stats),
        body,
    )


STYLE_CSS = """
:root {
  --bg: #f4f1ea;
  --panel: #fffdf9;
  --text: #1f2a2f;
  --muted: #5a666b;
  --border: #d9d2c6;
  --accent: #8b3d1f;
  --accent-soft: #f1e1d8;
  --ink-soft: #314248;
  --shadow: 0 18px 40px rgba(49, 66, 72, 0.08);
  --radius: 18px;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--text);
  background:
    radial-gradient(circle at top right, rgba(139, 61, 31, 0.08), transparent 30%),
    linear-gradient(180deg, #f7f4ed 0%, var(--bg) 100%);
  line-height: 1.65;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  font-family: Consolas, "Courier New", monospace;
  background: #f3efe7;
  padding: 0.12rem 0.35rem;
  border-radius: 6px;
}

.hero {
  padding: 54px 24px 34px;
  background:
    linear-gradient(135deg, rgba(139, 61, 31, 0.95), rgba(56, 73, 78, 0.94)),
    linear-gradient(180deg, #53301f, #2d3b40);
  color: #fffaf4;
}
.hero-inner {
  max-width: 1180px;
  margin: 0 auto;
}
.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.78rem;
  opacity: 0.84;
  margin-bottom: 10px;
}
.hero h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 3.6rem);
  line-height: 1.08;
}
.hero-lead {
  max-width: 920px;
  margin: 18px 0 12px;
  font-size: 1.08rem;
  color: rgba(255, 250, 244, 0.92);
}
.hero-meta {
  font-size: 0.94rem;
  color: rgba(255, 250, 244, 0.76);
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: 14px;
  margin-top: 28px;
}
.stat-card {
  background: rgba(255, 253, 249, 0.12);
  border: 1px solid rgba(255, 253, 249, 0.16);
  border-radius: 16px;
  padding: 18px 16px;
  backdrop-filter: blur(10px);
}
.stat-value {
  font-size: 1.8rem;
  font-weight: 700;
}
.stat-label {
  font-size: 0.92rem;
  margin-top: 4px;
}
.stat-note {
  font-size: 0.78rem;
  opacity: 0.8;
  margin-top: 6px;
}

.page-nav {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 14px 24px;
  background: rgba(255, 253, 249, 0.92);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(14px);
}
.page-nav a {
  display: inline-flex;
  align-items: center;
  padding: 9px 13px;
  border-radius: 999px;
  background: #f3ede2;
  color: var(--ink-soft);
  font-size: 0.92rem;
  text-decoration: none;
}
.page-nav a.active,
.page-nav a:hover {
  background: var(--accent-soft);
  color: var(--accent);
}

.container {
  max-width: 1180px;
  margin: 0 auto;
  padding: 30px 24px 70px;
}
.section {
  margin-top: 28px;
  padding: 28px;
  border-radius: var(--radius);
  background: var(--panel);
  box-shadow: var(--shadow);
  border: 1px solid rgba(217, 210, 198, 0.72);
}
.section-head {
  margin-bottom: 16px;
}
.section-kicker {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--accent);
  margin-bottom: 8px;
}
.section h2 {
  margin: 0;
  font-size: clamp(1.5rem, 2.6vw, 2.2rem);
}
.section h3 {
  margin: 0 0 12px;
  font-size: 1.18rem;
}
.section p {
  margin: 10px 0 0;
  color: var(--ink-soft);
}
.section-note {
  color: var(--muted);
}
.subsection {
  margin-top: 22px;
}

.card-grid,
.command-grid,
.download-grid,
.figure-grid {
  display: grid;
  gap: 18px;
  margin-top: 18px;
}
.card-grid {
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}
.command-grid {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}
.download-grid {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.figure-grid {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  align-items: start;
}

.info-card,
.command-card,
.download-card,
.figure-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: linear-gradient(180deg, #fffefa, #faf6ee);
  box-shadow: var(--shadow);
}
.info-card,
.command-card,
.download-card {
  padding: 18px;
}
.info-card h3,
.command-card h3,
.figure-card h3 {
  margin: 10px 0 8px;
}
.info-chip {
  display: inline-flex;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.78rem;
}
.info-meta {
  margin-top: 12px;
  font-size: 0.85rem;
  color: var(--muted);
}
.command-block,
.tree-block {
  margin: 12px 0 0;
  padding: 14px;
  overflow-x: auto;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: #fbf8f1;
  color: #2e393e;
  white-space: pre-wrap;
}

.note-box {
  margin-top: 20px;
  padding: 16px 18px;
  border-radius: 14px;
  background: #f6f0e8;
  border-left: 5px solid var(--accent);
  color: var(--ink-soft);
}

.table-wrap {
  margin-top: 12px;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  background: #fffdfa;
}
.data-table thead th {
  background: #efe7db;
  color: #263238;
  font-weight: 700;
}
.data-table th,
.data-table td {
  padding: 12px 14px;
  border-bottom: 1px solid #ece3d6;
  text-align: left;
  vertical-align: top;
  font-size: 0.94rem;
}
.data-table tbody tr:nth-child(even) {
  background: #fffcf7;
}

.figure-card {
  display: flex;
  flex-direction: column;
  padding: 16px;
}
.figure-card.compact {
  padding: 14px;
}
.image-frame {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 250px;
  aspect-ratio: 16 / 10;
  padding: 14px;
  background: #f6f1e8;
  border-radius: 14px;
  overflow: hidden;
}
.figure-card img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}
.figure-card figcaption {
  margin-top: 14px;
}
.figure-label {
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 6px;
}
.figure-card p {
  margin-top: 8px;
  font-size: 0.94rem;
  color: var(--muted);
}

.download-card {
  display: block;
  text-decoration: none;
  color: inherit;
}
.download-card:hover {
  transform: translateY(-2px);
  transition: transform 0.18s ease;
  text-decoration: none;
}
.download-type {
  color: var(--accent);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.download-name {
  margin-top: 8px;
  font-size: 1.05rem;
  font-weight: 700;
}
.download-note {
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.92rem;
}

.page-footer {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  padding: 24px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.9rem;
}

@media (max-width: 760px) {
  .hero {
    padding: 42px 18px 28px;
  }
  .page-nav {
    padding: 12px 16px;
  }
  .container {
    padding: 24px 16px 54px;
  }
  .section {
    padding: 20px;
  }
  .page-footer {
    flex-direction: column;
  }
}
"""


def write_static_files() -> None:
    ensure_dirs()
    (ASSETS_ROOT / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    readme_text = (
        "MARS-FIELD HTML Delivery Final\n"
        "============================\n\n"
        "Open index.html first.\n"
        "Other pages:\n"
        "- experiments.html : priority-ranked experiment page\n"
        "- runbook.html     : code entrypoints and run commands\n"
    )
    (OUT_ROOT / "README.txt").write_text(readme_text, encoding="utf-8")
    readme_md = (
        "# MARS-FIELD HTML Delivery Final\n\n"
        "当前交付包适合直接用于：\n\n"
        "- 项目总览汇报\n"
        "- GitHub 说明包\n"
        "- 论文素材导航页\n\n"
        "当前最准确的定位是：\n\n"
        "**MARS-FIELD engineering approximation v1**\n\n"
        "这意味着：\n\n"
        "- 它已经是可运行的研究原型平台\n"
        "- benchmark、图表、结构案例和 paper bundle 已经真实产出\n"
        "- 可以用于 GitHub 和论文交付素材整理\n"
        "- 但不应表述为 fully end-to-end、fully joint 的最终神经场模型\n"
    )
    (OUT_ROOT / "README.md").write_text(readme_md, encoding="utf-8")
    github_note = (
        "# GitHub / Paper Delivery Note\n\n"
        "推荐对外表述：\n\n"
        "- a benchmarked protein engineering research prototype\n"
        "- a field-style engineering approximation of MARS-FIELD\n"
        "- a codebase that already implements a unified evidence-to-sequence workflow\n\n"
        "不建议对外表述：\n\n"
        "- a completed fully neural end-to-end field model\n"
        "- a final production system\n"
        "- a fully joint generator-decoder-field foundation model\n"
    )
    (OUT_ROOT / "GITHUB_RELEASE_NOTE.md").write_text(github_note, encoding="utf-8")
    runtime_note = (
        "# Runtime Requirements Note\n\n"
        "当前仓库可以作为 **当前稳定研究原型** 直接交付，但它不是“任何新机器零配置一键全跑”的状态。\n\n"
        "当前能直接使用的部分：\n\n"
        "- HTML 交付包\n"
        "- 已生成的 benchmark / figure / structure 资产\n"
        "- 基于现有工作区数据的 Python 脚本调用\n\n"
        "当前仍依赖本地环境或外部组件的部分：\n\n"
        "- ProteinMPNN / ESM-IF 相关分支\n"
        "- IQ-TREE / MAFFT 等系统发育工具\n"
        "- AF3 服务器或外部结构预测流程\n"
        "- 若干脚本中仍引用本地路径或已有 outputs / datasets\n\n"
        "因此，当前最准确的说法是：\n\n"
        "- 作为 GitHub / 论文交付素材：可以直接交付\n"
        "- 作为当前工作区下的研究原型：可以直接运行主要流程\n"
        "- 作为跨机器零配置生产系统：还不是最终状态\n"
    )
    (OUT_ROOT / "RUNTIME_REQUIREMENTS.md").write_text(runtime_note, encoding="utf-8")


def write_pages() -> None:
    (OUT_ROOT / "index.html").write_text(build_index_html(), encoding="utf-8")
    (OUT_ROOT / "experiments.html").write_text(build_experiments_html(), encoding="utf-8")
    (OUT_ROOT / "runbook.html").write_text(build_runbook_html(), encoding="utf-8")


def check_links() -> dict[str, object]:
    results: dict[str, object] = {}
    for page_name in ["index.html", "experiments.html", "runbook.html"]:
        page_path = OUT_ROOT / page_name
        text = page_path.read_text(encoding="utf-8")
        refs = re.findall(r'(?:src|href)="([^"]+)"', text)
        missing = [
            ref for ref in refs
            if not ref.startswith("#") and not ref.startswith("http") and not (OUT_ROOT / ref).exists()
        ]
        results[page_name] = {"ref_count": len(refs), "missing": missing}
    return results


def write_manifest(link_report: dict[str, object]) -> None:
    manifest = {
        "package_version": "2.2-final",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(OUT_ROOT),
        "pages": ["index.html", "experiments.html", "runbook.html"],
        "zip_path": str(ZIP_PATH),
        "link_report": link_report,
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def make_zip() -> None:
    target_zip = ZIP_PATH
    if target_zip.exists():
        try:
            target_zip.unlink()
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_zip = target_zip.with_name(f"{target_zip.stem}_{stamp}{target_zip.suffix}")
    archive = shutil.make_archive(str(target_zip.with_suffix("")), "zip", OUT_ROOT.parent, OUT_ROOT.name)
    print(f"Wrote zip archive to {archive}")


def main() -> None:
    write_static_files()
    write_pages()
    link_report = check_links()
    write_manifest(link_report)
    make_zip()
    # Rebuild downloads so the zip itself is included on the pages.
    write_pages()
    link_report = check_links()
    write_manifest(link_report)
    print(f"Wrote HTML delivery package to {OUT_ROOT}")


if __name__ == "__main__":
    main()
