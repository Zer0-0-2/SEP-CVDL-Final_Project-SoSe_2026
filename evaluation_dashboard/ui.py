import gradio as gr
import pandas as pd
from config import evaluation_cache, CLASSES
from inference import analyze_image_all_models
from utils import get_model_list, format_model_name, get_experiment_folders, format_experiment_name
from config import logger, CLASSES, PROVIDED_CLASSES, evaluation_cache, evaluation_cache_provided
from gradio_gpu_monitor import GPUMonitor
import matplotlib.pyplot as plt
import seaborn as sns

def generate_confusion_matrix_plot(cm_data, is_provided=False):
    if not cm_data:
        return None
        
    labels_to_use = PROVIDED_CLASSES if is_provided else CLASSES
        
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f172a')
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='mako', 
                xticklabels=labels_to_use, yticklabels=labels_to_use, ax=ax,
                cbar_kws={'label': 'Count'})
    
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='#f8fafc', labelsize=10)
    ax.set_xlabel('Predicted Label', color='#f8fafc', fontsize=12)
    ax.set_ylabel('True Label', color='#f8fafc', fontsize=12)
    ax.set_title('Confusion Matrix', color='#f8fafc', fontsize=14)
    
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_tick_params(colors='#f8fafc')
        cbar.ax.yaxis.label.set_color('#f8fafc')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def update_leaderboard(experiment, show_filenames, sort_by, use_provided=False):
    logger.info(f"UI Event -> Updating leaderboard for stage: {experiment}")
    return get_leaderboard_df(experiment, show_filenames, sort_by, use_provided)

def update_model_dropdown(experiment, show_filenames):
    logger.info(f"UI Event -> Updating model dropdown choices for stage: {experiment}")
    if show_filenames:
        choices = [(w, w) for w in get_model_list(experiment)]
    else:
        choices = [(format_model_name(w), w) for w in get_model_list(experiment)]
    return gr.update(choices=choices, value=None)
def update_dashboard(weights_name, use_provided=False):
    cache = evaluation_cache_provided if use_provided else evaluation_cache
    
    if not weights_name or weights_name not in cache:
        return "N/A", "N/A", "N/A", "N/A", None, pd.DataFrame(), gr.update(choices=["All"], value="All"), []
        
    res = cache[weights_name]
    
    acc_str = f"{res['accuracy']:.2%}"
    f1_str = f"{res['macro_f1']:.2%}"
    prec_str = f"{res['macro_precision']:.2%}"
    rec_str = f"{res['macro_recall']:.2%}"
    
    df_metrics = res["df_metrics"]
    if "Support" not in df_metrics.columns:
        df_metrics["Support"] = "N/A"
        
    misclassifications = res["misclassifications"]
    cm_data = res.get("confusion_matrix", None)
    
    cm_plot = generate_confusion_matrix_plot(cm_data, use_provided)
    
    gallery_items = [(m["image_path"], f"True: {m['true_label']} | Pred: {m['pred_label']} (Conf: {m['confidence']:.2f})") for m in misclassifications]
    
    filter_classes = ["All"] + (PROVIDED_CLASSES if use_provided else CLASSES)
    
    return acc_str, prec_str, rec_str, f1_str, cm_plot, df_metrics, gr.update(choices=filter_classes, value="All"), gallery_items

def filter_errors(weights_name, true_class_filter, use_provided=False):
    cache = evaluation_cache_provided if use_provided else evaluation_cache
    if not weights_name or weights_name not in cache:
        return []
        
    misclassifications = cache[weights_name]["misclassifications"]
    
    if true_class_filter != "All":
        misclassifications = [m for m in misclassifications if m["true_label"] == true_class_filter]
        
    gallery_items = [(m["image_path"], f"True: {m['true_label']} | Pred: {m['pred_label']} (Conf: {m['confidence']:.2f})") for m in misclassifications]
    return gallery_items

def get_leaderboard_df(experiment="Initial_Experiments", show_filenames=False, sort_by="Accuracy", use_provided=False):
    cols = ["Stage", "Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    cache = evaluation_cache_provided if use_provided else evaluation_cache
    if not cache:
        df = pd.DataFrame(columns=cols)
        return df.rename(columns={sort_by: f"{sort_by} ▼"})
        
    allowed_models = set(get_model_list(experiment))
        
    from pathlib import Path
    data = []
    for w, res in cache.items():
        if w in allowed_models:
            model_name = w if show_filenames else format_model_name(w)
            
            experiment_name = str(Path(w).parent)
            if not show_filenames:
                experiment_name = format_experiment_name(experiment_name)
                
            data.append({
                "Stage": experiment_name,
                "Model": model_name,
                "Accuracy": res["accuracy"],
                "Precision": res["macro_precision"],
                "Recall": res["macro_recall"],
                "F1-Score": res["macro_f1"]
            })
    df = pd.DataFrame(data, columns=cols)
    
    if len(df) > 0:
        df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        
        df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{x:.2%}")
        df["Precision"] = df["Precision"].apply(lambda x: f"{x:.2%}")
        df["Recall"] = df["Recall"].apply(lambda x: f"{x:.2%}")
        df["F1-Score"] = df["F1-Score"].apply(lambda x: f"{x:.2%}")
            
    return df.rename(columns={sort_by: f"{sort_by} ▼"})

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
.metric-box .table-wrap,
.metric-box .tbody,
.metric-box .table-container,
.metric-box div[class*='overflow'],
.metric-box div,
.full-height-table .table-wrap,
.full-height-table .tbody,
.full-height-table .table-container,
.full-height-table div[class*='overflow'],
.full-height-table div {
    max-height: none !important;
    overflow: visible !important;
}
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
            experiment_folders = get_experiment_folders()
            experiment_choices = [("All Models", "All models")] + [(format_experiment_name(f), f) for f in experiment_folders]
            experiment_selector = gr.Radio(choices=experiment_choices, value="Initial_Experiments", label="Select Stage", interactive=True, scale=3)
            sort_selector = gr.Radio(choices=["Accuracy", "Precision", "Recall", "F1-Score"], value="Accuracy", label="Sort By", interactive=True, scale=3)
            with gr.Column(scale=3):
                show_filenames_toggle = gr.Checkbox(label="Show raw file names", value=False, interactive=True)
                provided_eval_toggle = gr.Checkbox(label="Use Provided Pipeline (YOLO+Dataset)", value=False, interactive=True)
        
        with gr.Tabs():
            with gr.TabItem("🏆 Leaderboard"):
                gr.Markdown("### 🌟 Overall Model Rankings")
                gr.Markdown("<p style='color: #94a3b8;'>Ranking of all available models based on overall metrics computed across our entire test dataset (Stanford Dogs, Oxford Pets and Imagenet)</p>")
                leaderboard_df_ui = gr.Dataframe(
                    interactive=False,
                    headers=["Stage", "Model", "Accuracy ▼", "Precision", "Recall", "F1-Score"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    elem_classes="metric-box",
                    column_widths=["15%", "45%", "10%", "10%", "10%", "10%"]
                )
                
            with gr.TabItem("📊 Dataset Evaluation"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=[(format_model_name(w), w) for w in get_model_list("Initial_Experiments")], 
                        label="Select Pretrained Weights", 
                        interactive=True, 
                        scale=3
                    )
                    eval_btn = gr.Button("🚀 Run Evaluation", variant="primary", scale=1)
                    
                gr.Markdown("### 📈 Headline Metrics")
                with gr.Row():
                    acc_box = gr.Textbox(label="Accuracy", interactive=False, elem_classes="metric-box")
                    prec_box = gr.Textbox(label="Precision", interactive=False, elem_classes="metric-box")
                    rec_box = gr.Textbox(label="Recall", interactive=False, elem_classes="metric-box")
                    f1_box = gr.Textbox(label="F1-Score", interactive=False, elem_classes="metric-box")
                    
                gr.Markdown("### 📋 Per-Class Performance Breakdown")
                df_box = gr.Dataframe(
                    interactive=False, 
                    label="Metrics by Breed",
                    headers=["Breed", "Precision", "Recall", "F1-Score", "Support"],
                    datatype=["str", "number", "number", "number", "number"],
                    elem_classes="full-height-table"
                )
                
                gr.Markdown("### 🔍 Confusion Matrix")
                conf_matrix_plot = gr.Plot()
                
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
                    inputs=[model_dropdown, provided_eval_toggle],
                    outputs=[acc_box, prec_box, rec_box, f1_box, conf_matrix_plot, df_box, error_filter, error_gallery]
                )
                
                error_filter.change(
                    fn=filter_errors,
                    inputs=[model_dropdown, error_filter, provided_eval_toggle],
                    outputs=[error_gallery]
                )
                
            with gr.TabItem("🖼️ Custom Upload"):
                gr.Markdown("### Test against all pretrained models")
                gr.Markdown("<p style='color: #94a3b8;'>Upload an image to run inference sequentially across all available weights.</p>")
                
                majority_vote_display = gr.Markdown("### 🗳️ Majority Vote: Waiting for inference...")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_img = gr.Image(type="filepath", label="Upload Image")
                        analyze_btn = gr.Button("🤖 Run Inference Against All Models", variant="primary")
                    with gr.Column(scale=2):
                        multi_model_df = gr.Dataframe(
                            label="Model Predictions",
                            interactive=False,
                            headers=["Model", "Prediction", "Confidence"],
                            datatype=["str", "str", "str"],
                            wrap=True
                        )
                    with gr.Column(scale=1):
                        yolo_result_img = gr.Image(type="numpy", label="YOLO Bounding Box Crop", interactive=False)
                        
                analyze_btn.click(
                    fn=analyze_image_all_models,
                    inputs=[upload_img, experiment_selector, show_filenames_toggle],
                    outputs=[multi_model_df, majority_vote_display, yolo_result_img]
                )
                
            with gr.TabItem("🖥️ System Monitor"):
                gr.Markdown("### 📊 Real-time Hardware Telemetry")
                gr.Markdown("<p style='color: #94a3b8;'>Monitor VRAM, Compute Utilization, and GPU Temperatures directly from the cluster.</p>")
                GPUMonitor()
                
            demo.load(fn=get_leaderboard_df, inputs=[experiment_selector, show_filenames_toggle, sort_selector, provided_eval_toggle], outputs=[leaderboard_df_ui])
            
            experiment_selector.change(
                fn=update_leaderboard,
                inputs=[experiment_selector, show_filenames_toggle, sort_selector, provided_eval_toggle],
                outputs=[leaderboard_df_ui]
            )
            experiment_selector.change(
                fn=update_model_dropdown,
                inputs=[experiment_selector, show_filenames_toggle],
                outputs=[model_dropdown]
            )
            
            show_filenames_toggle.change(
                fn=update_leaderboard,
                inputs=[experiment_selector, show_filenames_toggle, sort_selector, provided_eval_toggle],
                outputs=[leaderboard_df_ui]
            )
            show_filenames_toggle.change(
                fn=update_model_dropdown,
                inputs=[experiment_selector, show_filenames_toggle],
                outputs=[model_dropdown]
            )
            
            sort_selector.change(
                fn=update_leaderboard,
                inputs=[experiment_selector, show_filenames_toggle, sort_selector, provided_eval_toggle],
                outputs=[leaderboard_df_ui]
            )
            
            provided_eval_toggle.change(
                fn=update_leaderboard,
                inputs=[experiment_selector, show_filenames_toggle, sort_selector, provided_eval_toggle],
                outputs=[leaderboard_df_ui]
            )
            
            provided_eval_toggle.change(
                fn=update_dashboard,
                inputs=[model_dropdown, provided_eval_toggle],
                outputs=[acc_box, prec_box, rec_box, f1_box, conf_matrix_plot, df_box, error_filter, error_gallery]
            )
            
    return demo
