"""Django forms for the dashboard."""

from django import forms

from web.dashboard.models import CrawlJob


CHUNK_STRATEGY_CHOICES = [
    ("recursive", "Recursive (recommended)"),
    ("sentence", "Sentence-based"),
    ("semantic", "Semantic (paragraph-based)"),
    ("sliding_window", "Sliding Window"),
]

EXPORT_FORMAT_CHOICES = [
    ("jsonl", "JSONL"),
    ("parquet", "Parquet"),
    ("hf_dataset", "HuggingFace Dataset"),
]

LANGUAGE_CHOICES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
]


class CrawlJobForm(forms.Form):
    """Form for creating a new crawl job."""

    name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "placeholder": "e.g., Python Docs Crawl",
        }),
    )
    seed_urls = forms.CharField(
        widget=forms.Textarea(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm",
            "rows": 4,
            "placeholder": "https://docs.python.org/3/tutorial/\nhttps://wiki.python.org/",
        }),
        help_text="One URL per line",
    )
    max_pages = forms.IntegerField(
        min_value=1, max_value=100000, initial=100,
        widget=forms.NumberInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    max_depth = forms.IntegerField(
        min_value=0, max_value=20, initial=3,
        widget=forms.NumberInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    concurrency = forms.IntegerField(
        min_value=1, max_value=50, initial=5,
        widget=forms.NumberInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    delay_seconds = forms.FloatField(
        min_value=0, max_value=60, initial=1.0,
        widget=forms.NumberInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
            "step": "0.1",
        }),
    )
    respect_robots = forms.BooleanField(
        required=False, initial=True,
        widget=forms.CheckboxInput(attrs={
            "class": "h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500",
        }),
    )
    chunk_strategy = forms.ChoiceField(
        choices=CHUNK_STRATEGY_CHOICES, initial="recursive",
        widget=forms.Select(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    chunk_size = forms.IntegerField(
        min_value=64, max_value=8192, initial=512,
        widget=forms.NumberInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    chunk_overlap = forms.IntegerField(
        min_value=0, max_value=2048, initial=64,
        widget=forms.NumberInput(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    target_languages = forms.MultipleChoiceField(
        choices=LANGUAGE_CHOICES, initial=["en"], required=False,
        widget=forms.CheckboxSelectMultiple(attrs={
            "class": "h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500",
        }),
    )
    dedup_enabled = forms.BooleanField(
        required=False, initial=True,
        widget=forms.CheckboxInput(attrs={
            "class": "h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500",
        }),
    )
    export_format = forms.ChoiceField(
        choices=EXPORT_FORMAT_CHOICES, initial="jsonl",
        widget=forms.Select(attrs={
            "class": "w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent",
        }),
    )
    compress = forms.BooleanField(
        required=False, initial=False,
        widget=forms.CheckboxInput(attrs={
            "class": "h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500",
        }),
    )

    def clean_seed_urls(self):
        urls = self.cleaned_data["seed_urls"]
        url_list = [u.strip() for u in urls.strip().split("\n") if u.strip()]
        if not url_list:
            raise forms.ValidationError("At least one URL is required.")
        for url in url_list:
            if not url.startswith(("http://", "https://")):
                raise forms.ValidationError(f"Invalid URL: {url}. Must start with http:// or https://")
        return "\n".join(url_list)

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("chunk_overlap", 0) >= cleaned.get("chunk_size", 512):
            raise forms.ValidationError("Chunk overlap must be less than chunk size.")
        return cleaned

    def build_config_dict(self):
        """Convert form data to a PipelineConfig-compatible dict."""
        d = self.cleaned_data
        return {
            "crawl": {
                "seed_urls": d["seed_urls"].split("\n"),
                "max_pages": d["max_pages"],
                "max_depth": d["max_depth"],
                "concurrency": d["concurrency"],
                "delay_seconds": d["delay_seconds"],
                "respect_robots_txt": d["respect_robots"],
            },
            "extraction": {},
            "preprocess": {
                "target_languages": d.get("target_languages", ["en"]),
                "dedup_enabled": d.get("dedup_enabled", True),
            },
            "chunk": {
                "strategy": d["chunk_strategy"],
                "chunk_size": d["chunk_size"],
                "chunk_overlap": d["chunk_overlap"],
            },
            "embedding": {
                "enabled": False,
            },
            "export": {
                "format": d["export_format"],
                "output_dir": "output",
                "compress": d.get("compress", False),
            },
            "log_level": "INFO",
        }
