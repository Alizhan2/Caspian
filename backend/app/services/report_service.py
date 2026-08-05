import io
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FIELD_WARNING_EN = "This automated result indicates a potential oil-like surface anomaly. Field verification is required before any official environmental conclusion."
FIELD_WARNING_RU = "Автоматический результат указывает на потенциальную нефтеподобную аномалию поверхности. Перед официальными экологическими выводами необходима проверка на месте."
FIELD_WARNING_KK = "Автоматтандырылған нәтиже су бетіндегі мұнайға ұқсас ықтимал ауытқуды көрсетеді. Ресми экологиялық қорытындыға дейін далалық тексеру қажет."


def build_report_content(detection: dict, language: str) -> tuple[str, str]:
    risk_level = getattr(detection["risk_level"], "value", detection["risk_level"])
    if language == "ru":
        title = "Экологический скрининг Caspian Guardian AI"
        content = f"""{FIELD_WARNING_RU}\n\nИдентификатор обнаружения: {detection['id']}\nДата съёмки: {detection['acquisition_time']}\nПлощадь: {detection['area_km2']:.3f} км²\nУверенность: {detection['mean_confidence']:.0%} (максимум {detection['max_confidence']:.0%})\nРиск скрининга: {risk_level}\nКоординаты: {detection['coordinates']}\n\nМетодология: Sentinel-1 GRD VV/VH, нормализация SAR и сегментационная модель U-Net.\nОграничения: тёмные области SAR могут иметь разные причины и не являются подтверждением разлива. Результат предназначен для приоритизации полевой проверки.\nИсточник: Copernicus Sentinel data, processed by Caspian Guardian AI."""
    elif language == "kk":
        title = "Caspian Guardian AI экологиялық скринингі"
        content = f"""{FIELD_WARNING_KK}\n\nАнықтау идентификаторы: {detection['id']}\nТүсірілім күні: {detection['acquisition_time']}\nЕсептік аудан: {detection['area_km2']:.3f} км²\nСенімділік: {detection['mean_confidence']:.0%} (ең жоғары {detection['max_confidence']:.0%})\nСкрининг қаупі: {risk_level}\nКоординаттар: {detection['coordinates']}\n\nӘдістеме: Sentinel-1 GRD VV/VH, SAR қалыптандыруы және U-Net сегментация моделі.\nШектеулер: SAR-дегі қараңғы аймақтардың себептері әртүрлі болуы мүмкін және төгіндіні растамайды. Нәтиже далалық тексерудің басымдығын анықтауға арналған.\nДереккөз: Copernicus Sentinel деректері, Caspian Guardian AI өңдеуі."""
    else:
        title = "Caspian Guardian AI Environmental Screening"
        content = f"""{FIELD_WARNING_EN}\n\nDetection ID: {detection['id']}\nAcquisition date: {detection['acquisition_time']}\nEstimated area: {detection['area_km2']:.3f} km²\nConfidence: {detection['mean_confidence']:.0%} (maximum {detection['max_confidence']:.0%})\nScreening risk: {risk_level}\nCoordinates: {detection['coordinates']}\n\nMethodology: Sentinel-1 GRD VV/VH, SAR normalization, and U-Net segmentation.\nLimitations: dark SAR regions have multiple possible causes and do not confirm a spill. This result prioritizes field verification.\nSource: Copernicus Sentinel data, processed by Caspian Guardian AI."""
    return title, content


def _unicode_font() -> str:
    name = "GuardianUnicode"
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont(name, str(candidate)))
            return name
    return "Helvetica"


def _wrapped_lines(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def create_pdf(detection: dict, title: str, content: str, storage_path: str) -> Path:
    destination_dir = Path(storage_path).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"report-{detection['id']}.pdf"
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    font_name = _unicode_font()
    pdf.setFillColorRGB(0.11, 0.12, 0.09)
    pdf.rect(0, height - 95, width, 95, fill=1, stroke=0)
    pdf.setFillColorRGB(0.87, 0.66, 0.30)
    pdf.setFont(font_name, 18)
    pdf.drawString(42, height - 52, "CASPIAN GUARDIAN AI")
    pdf.setFillColorRGB(0.12, 0.13, 0.10)
    pdf.setFont(font_name, 14)
    y = height - 130
    for line in _wrapped_lines(title, font_name, 14, width - 84):
        pdf.drawString(42, y, line)
        y -= 18
    y -= 14
    pdf.setFont(font_name, 10)
    for paragraph in content.splitlines():
        for line in _wrapped_lines(paragraph, font_name, 10, width - 84):
            if y < 58:
                pdf.showPage()
                pdf.setFont(font_name, 10)
                y = height - 55
            pdf.drawString(42, y, line)
            y -= 15
        y -= 3
    pdf.setFillColorRGB(0.36, 0.40, 0.32)
    pdf.setFont(font_name, 8)
    pdf.drawString(42, 32, "Copernicus Sentinel | AI скрининг / AI скринингі | полевая / далалық проверка")
    pdf.save()
    destination.write_bytes(buffer.getvalue())
    return destination
