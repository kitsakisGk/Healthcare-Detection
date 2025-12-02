"""
PDF Report Generator for clinical diagnostic reports
Generates professional medical reports with predictions and explainability
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Union, List
from datetime import datetime
import io


class PDFReportGenerator:
    """
    Generate professional PDF reports for medical AI predictions
    """

    def __init__(self):
        """Initialize PDF generator"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                Table, TableStyle, PageBreak
            )
            from reportlab.pdfgen import canvas

            self.reportlab_available = True
            self.letter = letter
            self.A4 = A4
            self.inch = inch
            self.colors = colors
            self.getSampleStyleSheet = getSampleStyleSheet
            self.ParagraphStyle = ParagraphStyle
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.RLImage = RLImage
            self.Table = Table
            self.TableStyle = TableStyle
            self.PageBreak = PageBreak

        except ImportError:
            self.reportlab_available = False
            print("Warning: reportlab not installed. Install with: pip install reportlab")

    def generate_report(
        self,
        output_path: Union[str, Path],
        prediction: Dict,
        original_image: Union[np.ndarray, Image.Image, str, Path],
        gradcam_overlay: Optional[Union[np.ndarray, Image.Image]] = None,
        patient_info: Optional[Dict] = None,
        quality_check: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Generate comprehensive PDF report

        Args:
            output_path: Path to save PDF
            prediction: Prediction dictionary from Predictor
            original_image: Original X-ray image
            gradcam_overlay: Grad-CAM overlay image (optional)
            patient_info: Patient information dict (optional)
            quality_check: Quality control results (optional)
            metadata: Additional metadata (optional)
        """
        if not self.reportlab_available:
            raise RuntimeError("reportlab not installed")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF document
        doc = self.SimpleDocTemplate(
            str(output_path),
            pagesize=self.letter,
            rightMargin=0.75*self.inch,
            leftMargin=0.75*self.inch,
            topMargin=0.75*self.inch,
            bottomMargin=0.75*self.inch
        )

        # Build content
        story = []
        styles = self.getSampleStyleSheet()

        # Custom styles
        title_style = self.ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.colors.HexColor('#1f77b4'),
            spaceAfter=12,
            alignment=1  # Center
        )

        heading_style = self.ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=self.colors.HexColor('#2c3e50'),
            spaceAfter=8
        )

        # Header
        story.append(self.Paragraph("AI-POWERED PNEUMONIA DETECTION", title_style))
        story.append(self.Paragraph("Clinical Diagnostic Report", styles['Normal']))
        story.append(self.Spacer(1, 0.3*self.inch))

        # Report metadata
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(self.Paragraph(f"<b>Report Generated:</b> {report_time}", styles['Normal']))
        story.append(self.Spacer(1, 0.2*self.inch))

        # Patient Information (if provided)
        if patient_info:
            story.append(self.Paragraph("PATIENT INFORMATION", heading_style))
            patient_data = [
                ['Patient ID:', patient_info.get('id', 'N/A')],
                ['Age:', patient_info.get('age', 'N/A')],
                ['Sex:', patient_info.get('sex', 'N/A')],
                ['Study Date:', patient_info.get('study_date', 'N/A')]
            ]
            patient_table = self.Table(patient_data, colWidths=[2*self.inch, 4*self.inch])
            patient_table.setStyle(self.TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(patient_table)
            story.append(self.Spacer(1, 0.2*self.inch))

        # Prediction Results
        story.append(self.Paragraph("DIAGNOSTIC PREDICTION", heading_style))

        predicted_class = prediction.get('predicted_class', 'Unknown')
        confidence = prediction.get('confidence', 0.0) * 100
        risk_level = prediction.get('risk_level', 'Unknown')
        interpretation = prediction.get('confidence_interpretation', '')

        # Prediction color based on class
        if predicted_class == 'NORMAL':
            pred_color = self.colors.HexColor('#27ae60')
        else:
            pred_color = self.colors.HexColor('#e74c3c')

        pred_style = self.ParagraphStyle(
            'PredStyle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=pred_color,
            fontName='Helvetica-Bold'
        )

        story.append(self.Paragraph(f"Diagnosis: {predicted_class}", pred_style))
        story.append(self.Paragraph(f"Confidence: {confidence:.1f}% ({interpretation})", styles['Normal']))
        story.append(self.Paragraph(f"Risk Level: {risk_level.upper()}", styles['Normal']))
        story.append(self.Spacer(1, 0.2*self.inch))

        # Class Probabilities
        if 'probabilities' in prediction:
            story.append(self.Paragraph("Class Probabilities:", heading_style))
            class_names = ['NORMAL', 'BACTERIAL', 'VIRAL', 'COVID19']
            probs = prediction['probabilities']

            prob_data = [['Class', 'Probability']]
            for i, (cls, prob) in enumerate(zip(class_names, probs)):
                prob_data.append([cls, f"{prob*100:.2f}%"])

            prob_table = self.Table(prob_data, colWidths=[2*self.inch, 2*self.inch])
            prob_table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
                ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ]))
            story.append(prob_table)
            story.append(self.Spacer(1, 0.2*self.inch))

        # Uncertainty (if available)
        if 'uncertainty' in prediction:
            unc = prediction['uncertainty']
            story.append(self.Paragraph("Uncertainty Analysis:", heading_style))
            unc_data = [
                ['Epistemic Uncertainty:', f"{unc.get('epistemic', 0):.4f}"],
                ['Predictive Entropy:', f"{unc.get('entropy', 0):.4f}"]
            ]
            unc_table = self.Table(unc_data, colWidths=[2.5*self.inch, 1.5*self.inch])
            story.append(unc_table)
            story.append(self.Spacer(1, 0.2*self.inch))

        # Images
        story.append(self.Paragraph("MEDICAL IMAGES", heading_style))
        story.append(self.Spacer(1, 0.1*self.inch))

        # Original image
        if isinstance(original_image, (str, Path)):
            original_image = Image.open(original_image)
        elif isinstance(original_image, np.ndarray):
            original_image = Image.fromarray(original_image)

        # Save to temporary buffer
        img_buffer = io.BytesIO()
        original_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        story.append(self.Paragraph("<b>Original Chest X-Ray:</b>", styles['Normal']))
        story.append(self.RLImage(img_buffer, width=4*self.inch, height=4*self.inch))
        story.append(self.Spacer(1, 0.2*self.inch))

        # Grad-CAM overlay
        if gradcam_overlay is not None:
            if isinstance(gradcam_overlay, np.ndarray):
                gradcam_overlay = Image.fromarray(gradcam_overlay)

            gradcam_buffer = io.BytesIO()
            gradcam_overlay.save(gradcam_buffer, format='PNG')
            gradcam_buffer.seek(0)

            story.append(self.Paragraph("<b>Grad-CAM Explainability (Model Attention):</b>", styles['Normal']))
            story.append(self.RLImage(gradcam_buffer, width=4*self.inch, height=4*self.inch))
            story.append(self.Spacer(1, 0.2*self.inch))

        # Quality Control (if available)
        if quality_check:
            story.append(self.PageBreak())
            story.append(self.Paragraph("QUALITY CONTROL", heading_style))

            qc_status = "✓ PASSED" if quality_check.get('overall_pass', False) else "✗ FAILED"
            story.append(self.Paragraph(f"<b>Status:</b> {qc_status}", styles['Normal']))

            if 'checks' in quality_check:
                qc_data = [['Check', 'Status', 'Details']]
                for check_name, check_result in quality_check['checks'].items():
                    status = '✓ Pass' if check_result['pass'] else '✗ Fail'
                    msg = check_result.get('message', '')
                    qc_data.append([check_name.upper(), status, msg])

                qc_table = self.Table(qc_data, colWidths=[1.5*self.inch, 1*self.inch, 3.5*self.inch])
                qc_table.setStyle(self.TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, self.colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(qc_table)
                story.append(self.Spacer(1, 0.2*self.inch))

        # Disclaimer
        story.append(self.Spacer(1, 0.3*self.inch))
        disclaimer_style = self.ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=self.colors.HexColor('#7f8c8d'),
            leftIndent=0.5*self.inch,
            rightIndent=0.5*self.inch
        )

        disclaimer_text = """
        <b>MEDICAL DISCLAIMER:</b> This report is generated by an AI-powered diagnostic assistance system
        for research and educational purposes. The results should NOT be used as the sole basis for medical
        diagnosis or treatment decisions. Always consult qualified healthcare professionals for clinical
        interpretation and patient management. The AI system is a supplementary tool and does not replace
        professional medical judgment.
        """
        story.append(self.Paragraph(disclaimer_text, disclaimer_style))

        # Footer
        story.append(self.Spacer(1, 0.2*self.inch))
        footer_text = f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')} | " \
                     f"Generated by Healthcare Pneumonia Detection AI v2.0"
        story.append(self.Paragraph(footer_text, disclaimer_style))

        # Build PDF
        doc.build(story)

        print(f"✓ PDF report generated: {output_path}")

        return str(output_path)
