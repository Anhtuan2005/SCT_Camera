from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
qn = lambda tag: f"{{{W}}}{tag}"


REFERENCES = [
    "Linh Van Ma, Muhammad Ishfaq Hussain, JongHyun Park, Jeongbae Kim, và Moongu Jeon, “Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking,” arXiv:2312.01650, 2023.",
    "Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, và Xinggang Wang, “ByteTrack: Multi-Object Tracking by Associating Every Detection Box,” trong European Conference on Computer Vision (ECCV), Springer, tr. 1–21, 2022.",
    "Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, và Ben Upcroft, “Simple Online and Realtime Tracking,” trong IEEE International Conference on Image Processing (ICIP), tr. 3464–3468, 2016.",
    "Nicolai Wojke, Alex Bewley, và Dietrich Paulus, “Simple Online and Realtime Tracking with a Deep Association Metric,” trong IEEE International Conference on Image Processing (ICIP), tr. 3645–3649, 2017.",
    "Rudolph E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” Journal of Basic Engineering, quyển 82, số 1, tr. 35–45, 1960.",
    "Harold W. Kuhn, “The Hungarian Method for the Assignment Problem,” Naval Research Logistics Quarterly, quyển 2, số 1–2, tr. 83–97, 1955.",
    "Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, và Jie Chen, “DETRs Beat YOLOs on Real-Time Object Detection,” trong IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.",
    "Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, và Xi Shen, “DEIM: DETR with Improved Matching for Fast Convergence,” arXiv:2412.04234, 2025.",
    "Glenn Jocher, “Ultralytics YOLO11,” https://docs.ultralytics.com/models/yolo11/, truy cập năm 2026, 2024.",
    "Joseph Redmon, Santosh Divvala, Ross Girshick, và Ali Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” trong IEEE Conference on Computer Vision and Pattern Recognition (CVPR), tr. 779–788, 2016.",
    "Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, và Jian Sun, “YOLOX: Exceeding YOLO Series in 2021,” arXiv:2107.08430, 2021.",
    "G. Divya Deepak và Subraya Krishna Bhat, “Optimization of deep learning-based Faster R-CNN network for vehicle detection,” Scientific Reports, quyển 15, 38937, 2025.",
    "Jiankang Deng, Jia Guo, Niannan Xue, và Stefanos Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” trong IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.",
    "Kaiming He, Xiangyu Zhang, Shaoqing Ren, và Jian Sun, “Deep Residual Learning for Image Recognition,” trong IEEE Conference on Computer Vision and Pattern Recognition (CVPR), tr. 770–778, 2016.",
    "David W. Hosmer Jr., Stanley Lemeshow, và Rodney X. Sturdivant, “Applied Logistic Regression” (ấn bản thứ 3), John Wiley & Sons, 2013.",
    "Waqas Sultani, Chen Chen, và Mubarak Shah, “Real-World Anomaly Detection in Surveillance Videos,” trong IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), tr. 6479–6488, 2018.",
    "Bruce D. Lucas và Takeo Kanade, “An Iterative Image Registration Technique with an Application to Stereo Vision,” trong Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI), tr. 674–679, 1981.",
    "Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, và Konrad Schindler, “MOT16: A Benchmark for Multi-Object Tracking,” arXiv:1603.00831, 2016.",
]


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def set_single_text(paragraph: etree._Element, value: str) -> None:
    text_nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if len(text_nodes) != 1:
        raise RuntimeError(
            f"Expected exactly one text node, found {len(text_nodes)} in: {paragraph_text(paragraph)!r}"
        )
    text_nodes[0].text = value


def ensure_child(parent: etree._Element, name: str, first: bool = False) -> etree._Element:
    child = parent.find(qn(name))
    if child is None:
        child = etree.Element(qn(name))
        if first:
            parent.insert(0, child)
        else:
            parent.append(child)
    return child


def edit_document(xml_bytes: bytes) -> tuple[bytes, dict[str, object]]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)

    phone_paragraphs = [
        p
        for p in root.xpath(".//w:p", namespaces=NS)
        if "Điện thoại: [Bổ sung sau]" in paragraph_text(p)
    ]
    if len(phone_paragraphs) != 1:
        raise RuntimeError(f"Expected one phone placeholder paragraph, found {len(phone_paragraphs)}")
    old_phone = paragraph_text(phone_paragraphs[0])
    set_single_text(
        phone_paragraphs[0],
        old_phone.replace("Điện thoại: [Bổ sung sau]", "Điện thoại: 0977314645"),
    )

    reference_paragraphs = root.xpath(
        ".//w:p[w:pPr/w:pStyle[@w:val='TaiLieuThamKhao']]", namespaces=NS
    )
    if len(reference_paragraphs) != 18:
        raise RuntimeError(f"Expected 18 reference paragraphs, found {len(reference_paragraphs)}")
    old_references = [paragraph_text(p) for p in reference_paragraphs]
    for paragraph, reference in zip(reference_paragraphs, REFERENCES, strict=True):
        set_single_text(paragraph, reference)

    body_tables = root.xpath("./w:body/w:tbl", namespaces=NS)
    if len(body_tables) != 3:
        raise RuntimeError(f"Expected 3 body tables, found {len(body_tables)}")
    table_2 = body_tables[1]
    rows = table_2.xpath("./w:tr", namespaces=NS)
    if len(rows) < 2:
        raise RuntimeError("Table 2 has too few rows")

    cant_split_added = 0
    keep_next_added = 0
    for row_index, row in enumerate(rows):
        tr_pr = row.find(qn("trPr"))
        if tr_pr is None:
            tr_pr = etree.Element(qn("trPr"))
            row.insert(0, tr_pr)
        if tr_pr.find(qn("cantSplit")) is None:
            tr_pr.append(etree.Element(qn("cantSplit")))
            cant_split_added += 1

        if row_index < len(rows) - 1:
            for paragraph in row.xpath("./w:tc/w:p", namespaces=NS):
                p_pr = paragraph.find(qn("pPr"))
                if p_pr is None:
                    p_pr = etree.Element(qn("pPr"))
                    paragraph.insert(0, p_pr)
                if p_pr.find(qn("keepNext")) is None:
                    p_pr.append(etree.Element(qn("keepNext")))
                    keep_next_added += 1

    output = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    report = {
        "phone_before": old_phone,
        "phone_after": paragraph_text(phone_paragraphs[0]),
        "reference_count": len(reference_paragraphs),
        "old_references": old_references,
        "new_references": REFERENCES,
        "table_2_rows": len(rows),
        "cant_split_added": cant_split_added,
        "keep_next_added": keep_next_added,
    }
    return output, report


def copy_with_document_patch(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as zin:
        source_infos = zin.infolist()
        source_parts = {info.filename: zin.read(info.filename) for info in source_infos}

    patched_xml, report = edit_document(source_parts["word/document.xml"])

    with zipfile.ZipFile(destination, "w") as zout:
        for info in source_infos:
            cloned = copy.copy(info)
            data = patched_xml if info.filename == "word/document.xml" else source_parts[info.filename]
            zout.writestr(cloned, data)

    with zipfile.ZipFile(destination, "r") as zcheck:
        destination_parts = {name: zcheck.read(name) for name in zcheck.namelist()}
    changed_parts = [
        name for name in source_parts if source_parts[name] != destination_parts.get(name)
    ]
    if changed_parts != ["word/document.xml"]:
        raise RuntimeError(f"Unexpected changed package parts: {changed_parts}")
    if set(source_parts) != set(destination_parts):
        raise RuntimeError("Package part names changed")
    report["changed_package_parts"] = changed_parts
    report["output"] = str(destination)
    return report


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: edit_paper_three_issues.py SOURCE.docx OUTPUT.docx")
    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    if source == destination:
        raise RuntimeError("Destination must be a new file")
    if not source.exists():
        raise FileNotFoundError(source)
    report = copy_with_document_patch(source, destination)
    for key, value in report.items():
        if key not in {"old_references", "new_references"}:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
