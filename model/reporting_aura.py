# pylint: disable=too-many-arguments,too-many-branches,too-many-statements,too-many-locals
""" Relatórios de Treinamento da Aura """
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor

def save_tokens_with_rewards_to_pdf(
    input_ids,
    token_rewards,
    tokenizer,
    output_file="text.pdf",
    eps=0.2,
    eps2=0.5
):
    """ Salva os tokens com recompensas em PDF """

    c = canvas.Canvas(output_file, pagesize=letter)
    c.setFont("Courier", 8)
    x, y = 50, 750
    previous_text = ""
    current_text = ""

    for token_idx, reward in enumerate(token_rewards):
        current_text = tokenizer.decode(input_ids[: token_idx + 1])
        if current_text != previous_text:
            diff_text = current_text[len(previous_text) :]
            if "\n" in diff_text:
                lines = diff_text.split("\n")
                for line_idx, line in enumerate(lines):
                    if line_idx > 0:
                        x = 50
                        y -= 12
                    if abs(reward) < eps:
                        opacity = 0
                    elif abs(reward) > eps2:
                        opacity = 0.8
                    else:
                        opacity = 0.8 * (abs(reward) - eps) / (eps2 - eps)
                    text_width = c.stringWidth(line)
                    if reward > 0:
                        highlight_color = HexColor("#4CCD99")
                    else:
                        highlight_color = HexColor("#FFC700")
                    highlight_color.alpha = opacity
                    c.setFillColor(highlight_color)
                    c.rect(x, y - 2, text_width, 10, fill=True, stroke=False)
                    c.setFillColor(HexColor("#000000"))
                    c.drawString(x, y, line)
                    x += text_width
            else:
                if abs(reward) < eps:
                    opacity = 0
                elif abs(reward) > eps2:
                    opacity = 0.8
                else:
                    opacity = 0.8 * (abs(reward) - eps) / (eps2 - eps)
                text_width = c.stringWidth(diff_text)
                if reward > 0:
                    highlight_color = HexColor("#4CCD99")
                else:
                    highlight_color = HexColor("#FFC700")
                highlight_color.alpha = opacity
                c.setFillColor(highlight_color)
                c.rect(x, y - 2, text_width, 10, fill=True, stroke=False)
                c.setFillColor(HexColor("#000000"))
                c.drawString(x, y, diff_text)
                x += text_width
            if x > 550:
                x = 50
                y -= 12
            if y < 50:
                c.showPage()
                y = 750
                x = 50
            previous_text = current_text
    c.showPage()
    c.save()
