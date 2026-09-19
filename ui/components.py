from dash import html


def progress_bar(progress: int, stage: str):
    return html.Div([
        html.Div(stage, style={"marginBottom": "8px", "color": "#444", "fontSize": "14px"}),
        html.Div(
            style={
                "height": "12px",
                "backgroundColor": "#e9ecef",
                "borderRadius": "8px",
                "overflow": "hidden",
                "width": "100%",
            },
            children=[
                html.Div(
                    style={
                        "height": "12px",
                        "width": f"{max(0, min(100, progress))}%",
                        "background": "linear-gradient(90deg, #2f80ed, #56ccf2)",
                        "transition": "width 0.4s ease",
                    }
                )
            ],
        ),
        html.Div(f"{progress}%", style={"marginTop": "8px", "fontSize": "13px", "color": "#666"}),
    ])


def initial_result_placeholder():
    return html.Div(
        "暂无结果，请先在左侧输入需求，例如“为我绘制成都市土壤有机质图”。",
        style={
            "height": "100%",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "color": "#6c757d",
            "fontSize": "16px",
            "textAlign": "center",
            "padding": "24px",
            "border": "1px dashed #d0d7de",
            "borderRadius": "14px",
            "backgroundColor": "#fafbfc",
        },
    )


def title_block(title: str, subtitle: str):
    children = [html.H1(title, style={"margin": "0 0 8px 0", "fontSize": "28px"})]
    if subtitle:
        children.append(html.Div(subtitle, style={"color": "#6b7280", "fontSize": "14px"}))
    return html.Div(children, style={"marginBottom": "18px"})


def _rfk_result_card(text: str):
    sections = []
    current_title = None
    current_lines = []
    color_map = {
        "📊": ("#eff6ff", "#1d4ed8"),
        "⚙️": ("#f5f3ff", "#6d28d9"),
        "🤖": ("#fff7ed", "#c2410c"),
        "📁": ("#f8fafc", "#334155"),
    }
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        if line.startswith(("📊", "⚙️", "🤖", "📁")):
            if current_title or current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
        elif line.startswith("✅"):
            sections.append((line, []))
            current_title = None
            current_lines = []
        else:
            current_lines.append(line)
    if current_title or current_lines:
        sections.append((current_title, current_lines))

    children = []
    for title, lines in sections:
        title = title or ""
        if title.startswith("✅"):
            children.append(html.Div(title, style={"fontSize": "18px", "fontWeight": 900, "color": "#065f46", "marginBottom": "10px"}))
            continue
        icon = title[:2].strip() if title else ""
        bg, fg = color_map.get(icon, ("#f5f7fb", "#111827"))
        body = "\n".join([x for x in lines if x is not None]).strip()
        children.append(html.Div([
            html.Div(title, style={"fontSize": "16px", "fontWeight": 900, "color": fg, "marginBottom": "6px"}),
            html.Div(body, style={"whiteSpace": "pre-wrap", "lineHeight": "1.65", "fontSize": "14px", "color": "#111827"}) if body else None,
        ], style={"backgroundColor": bg, "border": f"1px solid {fg}22", "borderRadius": "14px", "padding": "10px 12px", "marginTop": "8px"}))
    return html.Div(children, style={"backgroundColor": "#ffffff", "padding": "12px 14px", "borderRadius": "18px", "maxWidth": "96%", "boxShadow": "0 3px 12px rgba(15,23,42,0.10)", "border": "1px solid #dbeafe"})


def chat_bubble(role: str, text: str):
    is_user = role == "user"
    bg = "#e8f0fe" if is_user else "#f5f7fb"
    align = "flex-end" if is_user else "flex-start"
    if (not is_user) and str(text or "").lstrip().startswith("✅ 土壤有机质制图完成"):
        inner = _rfk_result_card(str(text or ""))
    else:
        inner = html.Div(text, style={
            "backgroundColor": bg,
            "padding": "12px 14px",
            "borderRadius": "16px",
            "maxWidth": "88%",
            "whiteSpace": "pre-wrap",
            "lineHeight": "1.6",
            "fontSize": "15px",
            "color": "#111827",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
        })
    return html.Div(
        inner,
        style={"display": "flex", "justifyContent": align, "marginBottom": "10px"},
    )
