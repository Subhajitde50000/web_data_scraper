"""
Observability module
====================
[F17]  PrometheusMetrics   — /metrics endpoint, counters, gauges, histograms
[F18]  TracingMiddleware   — OpenTelemetry spans per URL (fetch → parse → store)
[F19]  GrafanaDashboard    — pre-built dashboard JSON, write to disk
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────
try:
    from prometheus_client import (
        Counter as PCounter, Gauge, Histogram,
        start_http_server as prom_start,
        CollectorRegistry, REGISTRY,
    )
    _PROM_OK = True
except ImportError:
    _PROM_OK = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.status import Status, StatusCode
    _OTEL_OK = True
except ImportError:
    _OTEL_OK = False


# ─────────────────────────────────────────────────────────────
# [F17] PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────

class PrometheusMetrics:
    """
    [F17] Prometheus metrics for PyScraper v7.

    Exposes /metrics on metrics_port (default 8000).
    All metric objects are None-safe — calling inc_*/set_*/observe_*
    when prometheus_client is not installed is a silent no-op.

    Metrics exposed:
      Counters:   pages_scraped_total, errors_total, duplicates_skipped_total
                  js_escalations_total, captcha_detected_total,
                  proxy_errors_total, circuit_open_total
      Gauges:     frontier_size, queue_depth, active_browsers,
                  proxy_pool_alive, domain_budget_remaining
      Histograms: fetch_duration_ms, parse_duration_ms, enrich_duration_ms
    """

    def __init__(self, port: int = 8000, enabled: bool = True):
        self._on = enabled and _PROM_OK
        if enabled and not _PROM_OK:
            log.warning("PrometheusMetrics: prometheus_client not installed — metrics disabled")
            log.warning("  pip install prometheus-client")
            return
        if not enabled:
            return

        self.pages_scraped = PCounter(
            "crawler_pages_scraped_total",
            "Pages successfully scraped",
            ["rendered_by", "language"],
        )
        self.errors = PCounter(
            "crawler_errors_total",
            "Fetch or parse errors",
            ["reason"],
        )
        self.duplicates = PCounter(
            "crawler_duplicates_skipped_total",
            "Duplicate pages skipped",
            ["kind"],               # exact / near / semantic
        )
        self.js_escalations = PCounter(
            "crawler_js_escalations_total",
            "Pages escalated from httpx to Playwright",
        )
        self.captcha_detected = PCounter(
            "crawler_captcha_detected_total",
            "CAPTCHA challenges detected",
        )
        self.proxy_errors = PCounter(
            "crawler_proxy_errors_total",
            "Proxy connection failures",
        )
        self.circuit_opens = PCounter(
            "crawler_circuit_open_total",
            "Domains blocked by circuit breaker",
            ["domain"],
        )

        self.frontier_size = Gauge(
            "crawler_frontier_size",
            "URLs pending in the frontier",
        )
        self.queue_depth = Gauge(
            "crawler_internal_queue_depth",
            "Items in fetch→parse queue",
        )
        self.active_browsers = Gauge(
            "crawler_active_browsers",
            "Playwright contexts currently open",
        )
        self.proxy_alive = Gauge(
            "crawler_proxy_pool_alive",
            "Number of healthy proxies",
        )
        self.budget_remaining = Gauge(
            "crawler_domain_budget_remaining",
            "Crawl budget remaining for a domain",
            ["domain"],
        )

        self.fetch_ms = Histogram(
            "crawler_fetch_duration_ms",
            "HTTP fetch latency in ms",
            buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )
        self.parse_ms = Histogram(
            "crawler_parse_duration_ms",
            "Parse + extraction latency in ms",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
        )
        self.enrich_ms = Histogram(
            "crawler_enrich_duration_ms",
            "NLP enrichment latency in ms",
            buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
        )

        prom_start(port)
        log.info("PrometheusMetrics: /metrics on port %d", port)

    # ── counters ──────────────────────────────────────────────

    def inc_scraped(self, rendered_by: str = "httpx", language: str = ""):
        if self._on: self.pages_scraped.labels(rendered_by=rendered_by, language=language or "unknown").inc()

    def inc_error(self, reason: str = "fetch"):
        if self._on: self.errors.labels(reason=reason).inc()

    def inc_duplicate(self, kind: str = "exact"):
        if self._on: self.duplicates.labels(kind=kind).inc()

    def inc_js_escalation(self):
        if self._on: self.js_escalations.inc()

    def inc_captcha(self):
        if self._on: self.captcha_detected.inc()

    def inc_proxy_error(self):
        if self._on: self.proxy_errors.inc()

    def inc_circuit_open(self, dom: str = ""):
        if self._on: self.circuit_opens.labels(domain=dom).inc()

    # ── gauges ────────────────────────────────────────────────

    def set_frontier(self, n: int):
        if self._on: self.frontier_size.set(n)

    def set_queue_depth(self, n: int):
        if self._on: self.queue_depth.set(n)

    def set_active_browsers(self, n: int):
        if self._on: self.active_browsers.set(n)

    def set_proxy_alive(self, n: int):
        if self._on: self.proxy_alive.set(n)

    def set_budget_remaining(self, dom: str, n: int):
        if self._on: self.budget_remaining.labels(domain=dom).set(n)

    # ── histograms ────────────────────────────────────────────

    def observe_fetch(self, ms: float):
        if self._on: self.fetch_ms.observe(ms)

    def observe_parse(self, ms: float):
        if self._on: self.parse_ms.observe(ms)

    def observe_enrich(self, ms: float):
        if self._on: self.enrich_ms.observe(ms)


# ─────────────────────────────────────────────────────────────
# [F18] OPENTELEMETRY TRACING
# ─────────────────────────────────────────────────────────────

class TracingMiddleware:
    """
    [F18] OpenTelemetry distributed tracing.

    Each URL produces a root span covering its full pipeline journey:
      crawler.fetch  → crawler.parse → crawler.enrich → crawler.store

    Child spans record timing for each stage. Exported to Jaeger / Tempo
    via OTLP gRPC (default: localhost:4317).

    Requires:
      pip install opentelemetry-sdk \\
                  opentelemetry-exporter-otlp-proto-grpc

    Run Jaeger locally:
      docker run -d --name jaeger \\
        -p 16686:16686 -p 4317:4317 \\
        jaegertracing/all-in-one:latest
    """

    def __init__(
        self,
        service_name:  str  = "pyscraper",
        otlp_endpoint: str  = "http://localhost:4317",
        enabled:       bool = True,
    ):
        self._enabled = enabled and _OTEL_OK
        self._tracer  = None

        if enabled and not _OTEL_OK:
            log.warning("TracingMiddleware: opentelemetry not installed — tracing disabled")
            log.warning("  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc")
            return
        if not enabled:
            return

        try:
            provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            log.info("TracingMiddleware: OTLP → %s", otlp_endpoint)
        except Exception as exc:
            log.warning("TracingMiddleware: init failed — %s", exc)
            self._enabled = False

    def start_url_span(self, url: str):
        """Return a root span for one URL's full journey. Use as context manager."""
        if not self._enabled or not self._tracer:
            return _NoopSpan()
        span = self._tracer.start_span("crawler.url")
        span.set_attribute("url", url)
        return span

    def child_span(self, parent_span, name: str, **attrs):
        """Create a child span under parent_span."""
        if not self._enabled or not self._tracer or isinstance(parent_span, _NoopSpan):
            return _NoopSpan()
        ctx = trace.set_span_in_context(parent_span)
        span = self._tracer.start_span(name, context=ctx)
        for k, v in attrs.items():
            span.set_attribute(k, str(v))
        return span

    def end_span(self, span, error: Optional[str] = None):
        if isinstance(span, _NoopSpan):
            return
        if error:
            span.set_status(Status(StatusCode.ERROR, error))
        span.end()


class _NoopSpan:
    """Silent no-op span when tracing is disabled."""
    def set_attribute(self, *a, **kw): pass
    def set_status(self, *a, **kw):    pass
    def end(self):                      pass
    def __enter__(self):                return self
    def __exit__(self, *a):             pass


# ─────────────────────────────────────────────────────────────
# [F19] GRAFANA DASHBOARD TEMPLATE
# ─────────────────────────────────────────────────────────────

GRAFANA_DASHBOARD = {
    "title": "PyScraper v7",
    "uid":   "pyscraper-v7",
    "schemaVersion": 38,
    "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"},
    "templating": {"list": []},
    "panels": [
        # ── Row 1: throughput ────────────────────────────────
        {
            "id": 1, "type": "stat", "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
            "title": "Pages / second",
            "targets": [{
                "expr": "rate(crawler_pages_scraped_total[1m])",
                "legendFormat": "{{rendered_by}}",
            }],
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
            "fieldConfig": {"defaults": {"unit": "reqps", "color": {"mode": "thresholds"},
                "thresholds": {"steps": [
                    {"color": "red", "value": 0},
                    {"color": "yellow", "value": 0.5},
                    {"color": "green", "value": 2},
                ]}}},
        },
        {
            "id": 2, "type": "stat", "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
            "title": "Total pages scraped",
            "targets": [{"expr": "sum(crawler_pages_scraped_total)", "legendFormat": "total"}],
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value"},
            "fieldConfig": {"defaults": {"unit": "short"}},
        },
        {
            "id": 3, "type": "stat", "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
            "title": "Error rate",
            "targets": [{"expr": "rate(crawler_errors_total[1m])", "legendFormat": "{{reason}}"}],
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
            "fieldConfig": {"defaults": {"unit": "reqps", "color": {"mode": "thresholds"},
                "thresholds": {"steps": [
                    {"color": "green", "value": 0},
                    {"color": "yellow", "value": 0.1},
                    {"color": "red", "value": 0.5},
                ]}}},
        },
        {
            "id": 4, "type": "stat", "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
            "title": "Frontier size",
            "targets": [{"expr": "crawler_frontier_size", "legendFormat": "pending"}],
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value"},
            "fieldConfig": {"defaults": {"unit": "short"}},
        },

        # ── Row 2: latency ───────────────────────────────────
        {
            "id": 5, "type": "timeseries", "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
            "title": "Fetch latency (p50 / p95 / p99)",
            "targets": [
                {"expr": "histogram_quantile(0.50, rate(crawler_fetch_duration_ms_bucket[5m]))",
                 "legendFormat": "p50"},
                {"expr": "histogram_quantile(0.95, rate(crawler_fetch_duration_ms_bucket[5m]))",
                 "legendFormat": "p95"},
                {"expr": "histogram_quantile(0.99, rate(crawler_fetch_duration_ms_bucket[5m]))",
                 "legendFormat": "p99"},
            ],
            "fieldConfig": {"defaults": {"unit": "ms"}},
        },
        {
            "id": 6, "type": "timeseries", "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
            "title": "Parse + Enrich latency (p50 / p95)",
            "targets": [
                {"expr": "histogram_quantile(0.50, rate(crawler_parse_duration_ms_bucket[5m]))",
                 "legendFormat": "parse p50"},
                {"expr": "histogram_quantile(0.95, rate(crawler_parse_duration_ms_bucket[5m]))",
                 "legendFormat": "parse p95"},
                {"expr": "histogram_quantile(0.95, rate(crawler_enrich_duration_ms_bucket[5m]))",
                 "legendFormat": "enrich p95"},
            ],
            "fieldConfig": {"defaults": {"unit": "ms"}},
        },

        # ── Row 3: infrastructure ────────────────────────────
        {
            "id": 7, "type": "timeseries", "gridPos": {"x": 0, "y": 12, "w": 8, "h": 6},
            "title": "Active Playwright browsers",
            "targets": [{"expr": "crawler_active_browsers", "legendFormat": "browsers"}],
            "fieldConfig": {"defaults": {"unit": "short", "min": 0}},
        },
        {
            "id": 8, "type": "timeseries", "gridPos": {"x": 8, "y": 12, "w": 8, "h": 6},
            "title": "Queue depth (fetch→parse)",
            "targets": [{"expr": "crawler_internal_queue_depth", "legendFormat": "depth"}],
            "fieldConfig": {"defaults": {"unit": "short", "min": 0}},
        },
        {
            "id": 9, "type": "timeseries", "gridPos": {"x": 16, "y": 12, "w": 8, "h": 6},
            "title": "Healthy proxies",
            "targets": [{"expr": "crawler_proxy_pool_alive", "legendFormat": "alive"}],
            "fieldConfig": {"defaults": {"unit": "short", "min": 0}},
        },

        # ── Row 4: dedup + languages ─────────────────────────
        {
            "id": 10, "type": "piechart", "gridPos": {"x": 0, "y": 18, "w": 8, "h": 7},
            "title": "Duplicate detection breakdown",
            "targets": [{"expr": "sum by (kind)(crawler_duplicates_skipped_total)",
                         "legendFormat": "{{kind}}"}],
        },
        {
            "id": 11, "type": "piechart", "gridPos": {"x": 8, "y": 18, "w": 8, "h": 7},
            "title": "Pages by language",
            "targets": [{"expr": "sum by (language)(crawler_pages_scraped_total)",
                         "legendFormat": "{{language}}"}],
        },
        {
            "id": 12, "type": "piechart", "gridPos": {"x": 16, "y": 18, "w": 8, "h": 7},
            "title": "Renderer breakdown (httpx vs playwright)",
            "targets": [{"expr": "sum by (rendered_by)(crawler_pages_scraped_total)",
                         "legendFormat": "{{rendered_by}}"}],
        },

        # ── Row 5: circuit breakers ───────────────────────────
        {
            "id": 13, "type": "table", "gridPos": {"x": 0, "y": 25, "w": 24, "h": 6},
            "title": "Circuit breaker trips by domain",
            "targets": [{"expr": "sort_desc(sum by (domain)(crawler_circuit_open_total))",
                         "legendFormat": "{{domain}}", "instant": True}],
            "fieldConfig": {"defaults": {"unit": "short"}},
        },
    ],
}


class GrafanaDashboard:
    """
    [F19] Generates a pre-built Grafana dashboard JSON for PyScraper metrics.

    Usage:
        GrafanaDashboard().save("pyscraper_dashboard.json")

    Then in Grafana: Dashboards → Import → Upload JSON file.
    Requires Prometheus data source named "Prometheus" (default).
    """

    def __init__(self, datasource_uid: str = "prometheus"):
        self._dash = json.loads(json.dumps(GRAFANA_DASHBOARD))  # deep copy
        # Inject datasource into all targets
        for panel in self._dash.get("panels", []):
            for target in panel.get("targets", []):
                target["datasource"] = {"type": "prometheus", "uid": datasource_uid}

    def save(self, path: str = "pyscraper_dashboard.json"):
        with open(path, "w") as f:
            json.dump({"dashboard": self._dash, "overwrite": True}, f, indent=2)
        log.info("GrafanaDashboard: saved to %s", path)
        log.info("  Import in Grafana: Dashboards → Import → Upload JSON file")
        return path

    def as_json(self) -> str:
        return json.dumps({"dashboard": self._dash, "overwrite": True}, indent=2)
