import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Image
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from .clustering import RollingClusterAnalyzer, ClusterAnalyzer
from .visualization import Visualizer
from .language import extract_lda_keywords

class ReportGenerator:
    """
    Handles the creation of PDF and HTML reports for cluster analysis.
    """
    def __init__(self, papers_library, bow_corpus, dictionary, run_dir, window_size=50, step_size=50, verbose=False, silhouette_scores=None, inertias=None, best_n_silhouette=None, best_n_elbow=None, num_clusters=None, model_tfidf=None, report_mode='pdf'):
        self.papers_library = papers_library
        self.bow_corpus = bow_corpus
        self.dictionary = dictionary
        self.run_dir = run_dir
        self.window_size = window_size
        self.step_size = step_size
        self.verbose = verbose
        self.silhouette_scores = silhouette_scores
        self.inertias = inertias
        self.best_n_silhouette = best_n_silhouette
        self.best_n_elbow = best_n_elbow
        self.num_clusters = num_clusters
        self.model_tfidf = model_tfidf
        self.report_mode = report_mode
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')

    def generate_report(self):
        """
        Generate the complete analysis report.
        """
        # Initialize ClusterTracker and RollingClusterAnalyzer
        analyzer = RollingClusterAnalyzer(window_size=self.window_size, step_size=self.step_size)
        
        # Debug window creation
        if self.verbose:
            print("\n=== Debugging Window Creation ===")
            analyzer.debug_window_creation(self.papers_library, verbose=self.verbose)
            print("=== End Debug Output ===\n")
        
        # Create windows for analysis
        analyzer.create_windows(self.papers_library, verbose=self.verbose)
        
        # Compute cluster frequencies
        freq_results = analyzer.compute_cluster_frequencies(self.papers_library.papers, verbose=self.verbose)
        self.frequencies = freq_results['frequencies']
        self.window_labels = freq_results['window_labels']
        self.cluster_ids = freq_results['cluster_ids']
        
        # Pre-compute all report data to avoid duplication
        report_data = self._compute_report_data()
        
        if self.report_mode == 'pdf':
            self._generate_pdf_report(report_data)
        else:
            self._generate_html_report(report_data)
        
        # Save cluster preferences to text file
        self._save_cluster_preferences(self.cluster_ids, self.frequencies)

    def _compute_report_data(self):
        """
        Compute all data needed for the report to avoid duplication between HTML and PDF.
        """
        report_data = {
            'cluster_overview': self._compute_cluster_overview_data(),
            'wcss_analysis': self._compute_wcss_data() if self.model_tfidf is not None else None,
            'plots': self._compute_plot_data()
        }
        return report_data

    def _compute_cluster_overview_data(self):
        """
        Compute cluster overview data (keywords, trends, sizes).
        """
        import html as html_escape
        
        cluster_sizes = pd.Series([p.cluster_id for p in self.papers_library.papers if p.cluster_id is not None]).value_counts().sort_index()
        cluster_data = []
        
        for cluster in self.cluster_ids:
            try:
                # Get documents in this cluster
                cluster_indices = [i for i, p in enumerate(self.papers_library.papers) if p.cluster_id == cluster]
                cluster_corpus = [self.bow_corpus[i] for i in cluster_indices]
                
                # Get keywords using LDA
                keywords = extract_lda_keywords(self.dictionary, cluster_corpus)
                keywords_str = ", ".join([k for k, _ in keywords])
                keywords_html = ", ".join([html_escape.escape(k) for k, _ in keywords])
                
                # Calculate trend
                trend = self._calculate_trend(cluster)
                
                cluster_data.append({
                    'cluster_id': cluster,
                    'size': cluster_sizes.get(cluster, 0),
                    'keywords': keywords_str,
                    'keywords_html': keywords_html,
                    'trend': trend
                })
                
            except Exception as e:
                print(f"Error processing cluster {cluster}: {e}")
                cluster_data.append({
                    'cluster_id': cluster,
                    'size': 0,
                    'keywords': 'Error',
                    'keywords_html': 'Error',
                    'trend': 'Error'
                })
        
        return cluster_data

    def _calculate_trend(self, cluster):
        """
        Calculate trend for a specific cluster.
        """
        cluster_idx = np.where(self.cluster_ids == cluster)[0][0]
        window_freqs = self.frequencies[:, cluster_idx]
        recent_n = min(5, len(window_freqs))
        past_n = min(15, len(window_freqs) - recent_n)
        
        if len(window_freqs) >= recent_n:
            recent = window_freqs[-recent_n:]
            if len(window_freqs) > past_n + recent_n:
                past = window_freqs[-past_n-recent_n:-recent_n]
            elif len(window_freqs) > recent_n:
                past = window_freqs[:-recent_n]
            else:
                past = window_freqs
            
            recent_mean = np.mean(recent)
            past_mean = np.mean(past) if len(past) > 0 else 0
            peak = np.max(window_freqs)
            
            if recent_mean < 0.04 * peak:
                return "Dormant"
            elif recent_mean > 1.5 * past_mean and recent_mean > 0.04:
                return "Trending"
            elif recent_mean < 0.5 * past_mean and past_mean > 0.04:
                return "Declining"
            else:
                return "Stable"
        else:
            return "Insufficient Data"

    def _compute_wcss_data(self):
        """
        Compute WCSS analysis data.
        """
        try:
            # Calculate WCSS
            tfidf_vectors = [self.model_tfidf[bow] for bow in self.bow_corpus]
            vocab_size = len(self.dictionary)
            tfidf_dense = np.zeros((len(tfidf_vectors), vocab_size))
            for i, vec in enumerate(tfidf_vectors):
                for idx, val in vec:
                    tfidf_dense[i, idx] = val
            
            cluster_labels = np.array([p.cluster_id for p in self.papers_library.papers])
            mean_wcss_per_cluster = ClusterAnalyzer.mean_wcss(tfidf_dense, cluster_labels, self.cluster_ids)
            
            # Transform for plotting
            plot_vals = 1 - np.array(mean_wcss_per_cluster)
            
            return {
                'cluster_ids': self.cluster_ids,
                'plot_vals': plot_vals,
                'ylabel': '1 - Mean Squared Distance to Centroid (log scale)',
                'title': 'Transformed Mean WCSS for Each Cluster'
            }
        except Exception as e:
            print(f"Error computing WCSS data: {e}")
            return None

    def _compute_plot_data(self):
        """
        Compute all plot data (silhouette/elbow, heatmap).
        """
        plots = {}
        
        # Silhouette and elbow plots
        if self.silhouette_scores and self.inertias and len(self.silhouette_scores) > 0 and len(self.inertias) > 0:
            try:
                visualizer = Visualizer()
                cluster_numbers = list(self.silhouette_scores.keys())
                scores = list(self.silhouette_scores.values())
                inertia_clusters = list(self.inertias.keys())
                inertia_values = list(self.inertias.values())
                
                buffer = visualizer.plot_silhouette_elbow(
                    cluster_numbers, scores, self.best_n_silhouette, self.num_clusters,
                    self.silhouette_scores, inertia_clusters, inertia_values, self.best_n_elbow, self.inertias
                )
                plots['silhouette_elbow'] = buffer
            except Exception as e:
                print(f"Error computing silhouette/elbow plots: {e}")
                plots['silhouette_elbow'] = None
        
        # WCSS plot
        if self.model_tfidf is not None:
            try:
                wcss_data = self._compute_wcss_data()
                if wcss_data:
                    visualizer = Visualizer()
                    buffer = visualizer.plot_mean_wcss(
                        wcss_data['cluster_ids'], wcss_data['plot_vals'],
                        wcss_data['ylabel'], wcss_data['title']
                    )
                    plots['wcss'] = buffer
                else:
                    plots['wcss'] = None
            except Exception as e:
                print(f"Error computing WCSS plot: {e}")
                plots['wcss'] = None
        
        # Heatmap
        try:
            visualizer = Visualizer()
            buffer = visualizer.plot_heatmap(self.frequencies, self.window_labels, self.cluster_ids)
            plots['heatmap'] = buffer
        except Exception as e:
            print(f"Error computing heatmap: {e}")
            plots['heatmap'] = None
        
        return plots

    def _generate_pdf_report(self, report_data):
        """
        Generate PDF report.
        """
        # Create PDF with landscape orientation
        doc = SimpleDocTemplate(
            os.path.join(self.run_dir, f"library_analysis_report_{self.timestamp}.pdf"),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Modern styles
        modern_title_style = ParagraphStyle(
            'ModernTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.HexColor("#22223b"),
            spaceAfter=18,
            keepWithNext=True,
        )
        modern_section_style = ParagraphStyle(
            'ModernSection',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor("#4a4e69"),
            backColor=colors.HexColor("#f2e9e4"),
            leftIndent=0,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        )
        modern_normal = ParagraphStyle(
            'ModernNormal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            textColor=colors.HexColor("#22223b"),
            spaceAfter=6,
        )

        # Title
        story.append(Paragraph("Library Analysis Report", modern_title_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8))
        
        # Cluster Overview
        story.extend(self._create_cluster_overview_section(report_data['cluster_overview'], modern_section_style, modern_normal))
        
        # Topic Identification Analysis
        story.extend(self._create_topic_identification_section(report_data['plots'], modern_section_style, modern_normal))
        
        # Per-Cluster Tightness (if model_tfidf is available)
        if report_data['wcss_analysis'] is not None:
            story.extend(self._create_wcss_section(report_data['wcss_analysis'], modern_section_style, modern_normal))
        
        # Topic Evolution Analysis
        story.extend(self._create_topic_evolution_section(report_data['plots'], modern_section_style, modern_normal))
        
        # Build PDF
        doc.build(story)
        print(f"PDF report saved to {self.run_dir}")

    def _generate_html_report(self, report_data):
        """
        Generate HTML report with actual content.
        """
        import html as html_escape
        
        html = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>Library Analysis Report</title>',
            '<style>',
            'body{font-family:Helvetica,Arial,sans-serif;background:#f8f9fa;color:#22223b;margin:0;padding:0;}',
            '.container{max-width:900px;margin:30px auto;background:#fff;border-radius:10px;box-shadow:0 2px 8px #ccc;padding:32px;}',
            'h1{color:#22223b;}',
            'h2{color:#4a4e69;background:#f2e9e4;padding:8px 12px;border-radius:6px;}',
            '.explainer{margin:8px 0 18px 0;color:#555;font-size:1.05em;}',
            'table{border-collapse:collapse;width:100%;margin-bottom:24px;}',
            'th,td{border:1px solid #c9ada7;padding:8px;text-align:left;}',
            'th{background:#4a4e69;color:#fff;}',
            'tr:nth-child(even){background:#f2e9e4;}',
            '.section{margin-bottom:40px;}',
            'img{display:block;margin:0 auto 12px auto;max-width:100%;border-radius:8px;box-shadow:0 1px 4px #bbb;}',
            'hr{border:none;border-top:1px solid #c9ada7;margin:32px 0;}',
            '.cluster-table{width:100%;}',
            '.cluster-table th{text-align:center;}',
            '.cluster-table td{text-align:center;}',
            '.error{color:#d32f2f;background:#ffebee;padding:10px;border-radius:5px;margin:10px 0;}',
            '.success{color:#388e3c;background:#e8f5e8;padding:10px;border-radius:5px;margin:10px 0;}',
            '</style>',
            '</head>',
            '<body>',
            '<div class="container">'
        ]
        
        html.append('<h1>Library Analysis Report</h1>')
        html.append(f'<p><strong>Generated:</strong> {self.timestamp}</p>')
        html.append('<hr>')
        
        # Cluster Overview Section
        html.append('<div class="section"><h2>Cluster Overview</h2>')
        html.append('<div class="explainer">This table summarizes each cluster, showing its size, top keywords, and trend over time. It gives a quick overview of the main topics in your library and how they are changing.</div>')
        
        try:
            # Create cluster overview table
            html.append('<table class="cluster-table">')
            html.append('<tr><th>Cluster</th><th>Number of Papers</th><th>Keywords</th><th>Trend</th></tr>')
            
            for cluster in report_data['cluster_overview']:
                html.append(f'<tr><td>Cluster {cluster["cluster_id"]}</td><td>{cluster["size"]}</td><td>{cluster["keywords_html"]}</td><td>{cluster["trend"]}</td></tr>')
            
            html.append('</table>')
            html.append('<div class="success">Cluster overview generated successfully.</div>')
            
        except Exception as e:
            html.append('<div class="error">Error generating cluster overview table.</div>')
            print(f"Error in cluster overview: {e}")
        
        html.append('</div><hr>')
        
        # Topic Identification Analysis Section
        html.append('<div class="section"><h2>Topic Identification Analysis</h2>')
        html.append('<div class="explainer">These plots show how well your data clusters for different numbers of clusters. The silhouette score (higher is better) and elbow method (lower is better) help you choose the best number of clusters for your data.</div>')
        
        try:
            if report_data['plots'].get('silhouette_elbow') is not None:
                # Convert plot to base64 for HTML embedding
                buffer = report_data['plots']['silhouette_elbow']
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode()
                html.append(f'<img src="data:image/png;base64,{plot_data}" alt="Silhouette and Elbow Plots">')
                html.append('<div class="success">Optimization plots generated successfully.</div>')
            else:
                html.append('<p>Optimization data not available.</p>')
                html.append('<div class="error">No silhouette or inertia data available for plotting.</div>')
                
        except Exception as e:
            html.append('<div class="error">Error generating optimization plots.</div>')
            print(f"Error in topic identification: {e}")
        
        html.append('</div><hr>')
        
        # Per-Cluster Tightness Section
        if report_data['wcss_analysis'] is not None:
            html.append('<div class="section"><h2>Per-Cluster Tightness</h2>')
            html.append('<div class="explainer">This bar chart shows 1 minus the mean within-cluster sum of squares (mean WCSS) for each cluster, on a log scale. Higher values indicate tighter, more specific clusters; lower values indicate more spread out or miscellaneous clusters.</div>')
            
            try:
                if report_data['plots'].get('wcss') is not None:
                    # Convert plot to base64 for HTML embedding
                    buffer = report_data['plots']['wcss']
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.read()).decode()
                    html.append(f'<img src="data:image/png;base64,{plot_data}" alt="WCSS Analysis">')
                    html.append('<div class="success">WCSS analysis generated successfully.</div>')
                else:
                    html.append('<div class="error">WCSS analysis not available.</div>')
                
            except Exception as e:
                html.append('<div class="error">Error generating WCSS analysis.</div>')
                print(f"Error in WCSS analysis: {e}")
            
            html.append('</div><hr>')
        
        # Topic Evolution Analysis Section
        html.append('<div class="section"><h2>Topic Evolution Analysis</h2>')
        html.append('<div class="explainer">This heatmap shows how the distribution of clusters changes over time. Each row is a cluster, and each column is a time window. It helps you see which topics are emerging, stable, or fading in your library.</div>')
        
        try:
            if report_data['plots'].get('heatmap') is not None:
                # Convert plot to base64 for HTML embedding
                buffer = report_data['plots']['heatmap']
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.read()).decode()
                html.append(f'<img src="data:image/png;base64,{plot_data}" alt="Cluster Evolution Heatmap">')
                html.append('<div class="success">Topic evolution heatmap generated successfully.</div>')
            else:
                html.append('<div class="error">Topic evolution heatmap not available.</div>')
            
        except Exception as e:
            html.append('<div class="error">Error generating topic evolution heatmap.</div>')
            print(f"Error in topic evolution: {e}")
        
        html.append('</div>')
        html.append('<hr>')
        html.append('<div class="section">')
        html.append('<h2>Report Summary</h2>')
        html.append(f'<p><strong>Total Papers:</strong> {len(self.papers_library.papers)}</p>')
        html.append(f'<p><strong>Number of Clusters:</strong> {len(self.cluster_ids)}</p>')
        html.append(f'<p><strong>Number of Time Windows:</strong> {len(self.window_labels)}</p>')
        html.append('</div>')
        html.append('</div></body></html>')
        
        html_path = os.path.join(self.run_dir, f"library_analysis_report_{self.timestamp}.html")
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html))
            print(f"HTML report saved to {html_path}")
        except Exception as e:
            print(f"Error saving HTML report: {e}")
            # Try saving a simplified version
            simple_html = [
                '<!DOCTYPE html>',
                '<html><head><meta charset="utf-8"><title>Library Analysis Report</title></head>',
                '<body><h1>Library Analysis Report</h1>',
                '<p>Error generating full report. Please check the console output for details.</p>',
                '</body></html>'
            ]
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(simple_html))

    def _create_cluster_overview_section(self, cluster_data, modern_section_style, modern_normal):
        """
        Create the cluster overview section for PDF.
        """
        story = []
        
        # Create table data
        table_data = [['Cluster', 'Number of Papers', 'Keywords', 'Trend']]
        
        for cluster in cluster_data:
            # Add row to table
            table_data.append([
                f'Cluster {cluster["cluster_id"]}',
                str(cluster["size"]),
                cluster["keywords"],
                cluster["trend"]
            ])
        
        # Create table
        t = Table(table_data, colWidths=[1.2*inch, 1.5*inch, 3.5*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a4e69")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f2e9e4")),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#22223b")),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#c9ada7")),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        # Cluster Overview explainer
        cluster_overview_explainer = Paragraph(
            "<b>What is this?</b> This table summarizes each cluster, showing its size, top keywords, and trend over time. "
            "<b>Why look at it?</b> It gives a quick overview of the main topics in your library and how they are changing.",
            modern_normal)
        
        story.extend([
            Paragraph("Cluster Overview", modern_section_style),
            cluster_overview_explainer,
            Spacer(1, 6),
            t,
            Spacer(1, 8),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
        ])
        
        return story

    def _create_topic_identification_section(self, plots, modern_section_style, modern_normal):
        """
        Create the topic identification section for PDF.
        """
        story = []
        
        if plots.get('silhouette_elbow') is not None:
            # Topic Identification Analysis explainer
            silhouette_explainer = Paragraph(
                "<b>What is this?</b> These plots show how well your data clusters for different numbers of clusters. "
                "The left plot shows the silhouette score (higher is better), and the right shows the elbow method (lower is better). "
                "<b>Why look at it?</b> It helps you choose the best number of clusters for your data.",
                modern_normal)
            
            story.extend([
                Paragraph("Topic Identification Analysis", modern_section_style),
                silhouette_explainer,
                Spacer(1, 6),
                Image(plots['silhouette_elbow'], width=500, height=180),
                Spacer(1, 8),
                HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
            ])
        
        return story

    def _create_wcss_section(self, wcss_data, modern_section_style, modern_normal):
        """
        Create the WCSS section for PDF.
        """
        story = []
        
        # Create plot
        visualizer = Visualizer()
        buffer = visualizer.plot_mean_wcss(
            wcss_data['cluster_ids'], wcss_data['plot_vals'],
            wcss_data['ylabel'], wcss_data['title']
        )
        
        wcss_explainer = Paragraph(
            "<b>What is this?</b> This bar chart shows 1 minus the mean within-cluster sum of squares (mean WCSS) for each cluster, on a log scale. "
            "<b>Why look at it?</b> Higher values indicate tighter, more specific clusters; lower values indicate more spread out or miscellaneous clusters. The log scale helps visualize differences when values are close to 1.",
            modern_normal)
        
        story.extend([
            Paragraph("Per-Cluster Tightness (1 - Mean WCSS, log scale)", modern_section_style),
            wcss_explainer,
            Spacer(1, 6),
            Image(buffer, width=500, height=180),
            Spacer(1, 8),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
        ])
        
        return story

    def _create_topic_evolution_section(self, plots, modern_section_style, modern_normal):
        """
        Create the topic evolution section for PDF.
        """
        story = []
        
        if plots.get('heatmap') is not None:
            te_explainer = Paragraph(
                "<b>What is this?</b> This heatmap shows how the distribution of clusters changes over time. "
                "Each row is a cluster, and each column is a time window. "
                "<b>Why look at it?</b> It helps you see which topics are emerging, stable, or fading in your library.",
                modern_normal)
            
            story.extend([
                Paragraph("Topic Evolution Analysis", modern_section_style),
                te_explainer,
                Spacer(1, 6),
                Image(plots['heatmap'], width=500, height=180),
                Spacer(1, 8),
                HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c9ada7"), spaceBefore=8, spaceAfter=8)
            ])
        
        return story

    def _save_cluster_preferences(self, cluster_ids, frequencies):
        """
        Save cluster preferences to text file.
        """
        cluster_lines = []
        cluster_sizes = pd.Series([p.cluster_id for p in self.papers_library.papers if p.cluster_id is not None]).value_counts().sort_index()
        
        for cluster in cluster_ids:
            # Get documents in this cluster
            cluster_indices = [i for i, p in enumerate(self.papers_library.papers) if p.cluster_id == cluster]
            cluster_corpus = [self.bow_corpus[i] for i in cluster_indices]
            
            # Get keywords using LDA
            keywords = extract_lda_keywords(self.dictionary, cluster_corpus)
            keywords_str = ", ".join([k for k, _ in keywords])
            
            cluster_lines.append(f"Cluster {cluster}: {cluster_sizes[cluster]} papers - {keywords_str}")
        
        # Save cluster preferences to text file
        preferences_file = os.path.join(self.run_dir, 'feed_search_preferences.txt')
        with open(preferences_file, 'w') as f:
            f.write('\n'.join(cluster_lines))
        
        print(f"Feed preferences saved to {preferences_file}") 