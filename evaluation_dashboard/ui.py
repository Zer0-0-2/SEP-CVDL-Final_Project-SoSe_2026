import gradio as gr
import pandas as pd
from config import evaluation_cache, CLASSES
from inference import analyze_image_all_models
from utils import get_model_list, format_model_name, get_experiment_folders
from gradio_gpu_monitor import GPUMonitor

def update_leaderboard(experiment, show_filenames):
    return get_leaderboard_df(experiment, show_filenames)

def update_model_dropdown(experiment, show_filenames):
    if show_filenames:
        choices = [(w, w) for w in get_model_list(experiment)]
    else:
        choices = [(format_model_name(w), w) for w in get_model_list(experiment)]
    return gr.update(choices=choices, value=None)
def update_dashboard(weights_name):
    if not weights_name or weights_name not in evaluation_cache:
        return "N/A", "N/A", "N/A", "N/A", pd.DataFrame(), gr.update(choices=["All"], value="All"), []
        
    res = evaluation_cache[weights_name]
    
    acc_str = f"{res['accuracy']:.2%}"
    f1_str = f"{res['macro_f1']:.2%}"
    prec_str = f"{res['macro_precision']:.2%}"
    rec_str = f"{res['macro_recall']:.2%}"
    
    df_metrics = res["df_metrics"]
    misclassifications = res["misclassifications"]
    
    gallery_items = [(m["image_path"], f"Pred: {m['pred_label']} (Conf: {m['confidence']:.2f})") for m in misclassifications]
    
    return acc_str, f1_str, prec_str, rec_str, df_metrics, gr.update(choices=["All"] + CLASSES, value="All"), gallery_items

def filter_errors(weights_name, true_class_filter):
    if not weights_name or weights_name not in evaluation_cache:
        return []
        
    misclassifications = evaluation_cache[weights_name]["misclassifications"]
    
    if true_class_filter != "All":
        misclassifications = [m for m in misclassifications if m["true_label"] == true_class_filter]
        
    gallery_items = [(m["image_path"], f"Pred: {m['pred_label']} (Conf: {m['confidence']:.2f})") for m in misclassifications]
    return gallery_items

def get_leaderboard_df(experiment="All models", show_filenames=False):
    if not evaluation_cache:
        return pd.DataFrame(columns=["Model", "Accuracy", "F1-Score", "Precision", "Recall"])
        
    allowed_models = set(get_model_list(experiment))
        
    data = []
    for w, res in evaluation_cache.items():
        if w in allowed_models:
            model_name = w if show_filenames else format_model_name(w)
            data.append({
                "Model": model_name,
                "Accuracy": res["accuracy"],
                "F1-Score": res["macro_f1"],
                "Precision": res["macro_precision"],
                "Recall": res["macro_recall"]
            })
    df = pd.DataFrame(data)
    
    df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    
    df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{x:.2%}")
    df["F1-Score"] = df["F1-Score"].apply(lambda x: f"{x:.2%}")
    df["Precision"] = df["Precision"].apply(lambda x: f"{x:.2%}")
    df["Recall"] = df["Recall"].apply(lambda x: f"{x:.2%}")
    
    if len(df) > 0:
        df.loc[0, 'Model'] = "🥇 " + df.loc[0, 'Model']
    if len(df) > 1:
        df.loc[1, 'Model'] = "🥈 " + df.loc[1, 'Model']
    if len(df) > 2:
        df.loc[2, 'Model'] = "🥉 " + df.loc[2, 'Model']
        
    return df

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body {
    background-color: #0f172a !important;
    background-image: radial-gradient(circle at top center, #1e293b, #0f172a 80%) !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.metric-box {
    text-align: center;
    padding: 24px;
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.metric-box input {
    background: transparent !important;
    border: none !important;
    color: #f8fafc !important;
    text-align: center;
    font-weight: 600;
}
h1 {
    background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.tab-nav {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.tabitem {
    border: none !important;
    background: transparent !important;
}

/* Force tables to expand without scrollbars */
.table-wrap { max-height: none !important; overflow: visible !important; }
.tbody { max-height: none !important; overflow: visible !important; }
.table-container { max-height: none !important; overflow: visible !important; }
"""

my_theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="rgba(30, 41, 59, 0.5)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.1)",
    border_color_primary="rgba(255,255,255,0.1)",
    background_fill_primary="#0f172a",
    button_primary_background_fill="linear-gradient(135deg, #3b82f6, #8b5cf6)",
    button_primary_background_fill_hover="linear-gradient(135deg, #2563eb, #7c3aed)"
)

def create_ui():
    with gr.Blocks(title="Cat and Dog Breed Classifier Dashboard") as demo:
        gr.HTML("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem;'>🐾 Cat and Dog Breed Classifier Dashboard</h1>")
        gr.HTML("<p style='text-align: center; color: #94a3b8; margin-bottom: 2rem;'>SEP CVDL Catfish Coders</p>")
        
        with gr.Row():
            experiment_choices = ["All models"] + get_experiment_folders()
            experiment_selector = gr.Radio(choices=experiment_choices, value="All models", label="Select Experiment", interactive=True, scale=3)
            show_filenames_toggle = gr.Checkbox(label="Show raw file names", value=False, interactive=True, scale=1)
        
        with gr.Tabs():
            with gr.TabItem("🏆 Leaderboard"):
                gr.Markdown("### 🌟 Overall Model Rankings")
                gr.Markdown("<p style='color: #94a3b8;'>Ranking of all available models based on overall metrics computed across our entire test dataset (Stanford Dogs, Oxford Pets and Imagenet)</p>")
                leaderboard_df_ui = gr.Dataframe(
                    interactive=False,
                    headers=["Model", "Accuracy", "F1-Score", "Precision", "Recall"],
                    datatype=["str", "str", "str", "str", "str"],
                    elem_classes="metric-box"
                )
                
            with gr.TabItem("📊 Dataset Evaluation"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=[(format_model_name(w), w) for w in get_model_list()], 
                        label="Select Pretrained Weights", 
                        interactive=True, 
                        scale=3
                    )
                    eval_btn = gr.Button("🚀 Run Evaluation", variant="primary", scale=1)
                    
                gr.Markdown("### 📈 Headline Metrics")
                with gr.Row():
                    acc_box = gr.Textbox(label="Accuracy", interactive=False, elem_classes="metric-box")
                    f1_box = gr.Textbox(label="F1-Score", interactive=False, elem_classes="metric-box")
                    prec_box = gr.Textbox(label="Precision", interactive=False, elem_classes="metric-box")
                    rec_box = gr.Textbox(label="Recall", interactive=False, elem_classes="metric-box")
                    
                gr.Markdown("### 📋 Per-Class Performance Breakdown")
                df_box = gr.Dataframe(
                    interactive=False, 
                    label="Metrics by Breed",
                    headers=["Breed", "Precision", "Recall", "F1-Score"],
                    datatype=["str", "number", "number", "number"],
                )
                
                gr.Markdown("### 🔍 Visual Error Explorer")
                gr.Markdown("<p style='color: #94a3b8;'>Explore misclassified images. Select a True Class below to filter.</p>")
                with gr.Row():
                    error_filter = gr.Dropdown(choices=["All"], value="All", label="Filter Errors by True Class", interactive=True)
                
                error_gallery = gr.Gallery(
                    label="Misclassifications", 
                    show_label=True, 
                    columns=4, 
                    height="auto",
                    object_fit="contain",
                    interactive=False
                )
                
                eval_btn.click(
                    fn=update_dashboard,
                    inputs=[model_dropdown],
                    outputs=[acc_box, f1_box, prec_box, rec_box, df_box, error_filter, error_gallery]
                )
                
                error_filter.change(
                    fn=filter_errors,
                    inputs=[model_dropdown, error_filter],
                    outputs=[error_gallery]
                )
                
            with gr.TabItem("🖼️ Custom Upload"):
                gr.Markdown("### Test against all pretrained models")
                gr.Markdown("<p style='color: #94a3b8;'>Upload an image to run inference sequentially across all available weights.</p>")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_img = gr.Image(type="numpy", label="Upload Image")
                        analyze_btn = gr.Button("🤖 Run Inference Against All Models", variant="primary")
                    with gr.Column(scale=2):
                        multi_model_df = gr.Dataframe(
                            label="Model Predictions",
                            interactive=False,
                            headers=["Model", "Prediction", "Confidence"],
                            datatype=["str", "str", "str"]
                        )
                        
                analyze_btn.click(
                    fn=analyze_image_all_models,
                    inputs=[upload_img, experiment_selector, show_filenames_toggle],
                    outputs=[multi_model_df]
                )
                
            with gr.TabItem("🖥️ System Monitor"):
                gr.Markdown("### 📊 Real-time Hardware Telemetry")
                gr.Markdown("<p style='color: #94a3b8;'>Monitor VRAM, Compute Utilization, and GPU Temperatures directly from the cluster.</p>")
                GPUMonitor()
                
            demo.load(fn=get_leaderboard_df, inputs=[experiment_selector, show_filenames_toggle], outputs=[leaderboard_df_ui])
            
            experiment_selector.change(
                fn=update_leaderboard,
                inputs=[experiment_selector, show_filenames_toggle],
                outputs=[leaderboard_df_ui]
            )
            experiment_selector.change(
                fn=update_model_dropdown,
                inputs=[experiment_selector, show_filenames_toggle],
                outputs=[model_dropdown]
            )
            
            show_filenames_toggle.change(
                fn=update_leaderboard,
                inputs=[experiment_selector, show_filenames_toggle],
                outputs=[leaderboard_df_ui]
            )
            show_filenames_toggle.change(
                fn=update_model_dropdown,
                inputs=[experiment_selector, show_filenames_toggle],
                outputs=[model_dropdown]
            )
            
    return demo
