# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PagerDuty alerting helpers: trigger and resolve incidents via the Events V2 API.
"""
import pypd

import bittensor as bt

def triggerPagerDutyIncident(integration_keys, source, group, event_class, msg, custom_details=None, severity="error", dedup_key=None):
    """
    Trigger a PagerDuty incident via the Events V2 API for each integration key.

    Args:
        integration_keys (list[str]): PagerDuty routing keys to send the alert to.
        source (str): Component or service that generated the event.
        group (str): Logical grouping for the incident (e.g. subnet name).
        event_class (str): Class of the event (e.g. 'validator', 'simulator').
        msg (str): Human-readable summary of the incident.
        custom_details (dict, optional): Additional structured context for the alert.
        severity (str): Alert severity — 'critical', 'error', 'warning', or 'info'.
            Defaults to 'error'.
        dedup_key (str, optional): Key used to deduplicate or correlate incidents.
    """
    if integration_keys and len(integration_keys) > 0:
        bt.logging.error(msg)
        for integration_key in integration_keys:
            if integration_key is None: continue
            try:
                data={
                    'routing_key': integration_key,
                    'event_action': 'trigger',
                    'dedup_key' : dedup_key,
                    'payload': {
                        'summary': msg,
                        'severity': severity,
                        'source': source,
                        'class' : event_class,
                        'group' : group
                    }
                }
                if custom_details is not None:
                    data['payload']['custom_details'] = custom_details
                pypd.EventV2.create(data=data)
            except Exception as e:
                bt.logging.error(f"FAILED TO GENERATE PAGERDUTY ALERT : {str(e)}\n{data if data else ''}")

def resolvePagerDutyIncident(integration_keys, source, dedup_key):
    """
    Resolve an existing PagerDuty incident via the Events V2 API.

    Args:
        integration_keys (list[str]): PagerDuty routing keys for the integration.
        source (str): Component or service resolving the incident.
        dedup_key (str): Deduplication key identifying the incident to resolve.
    """
    if integration_keys and len(integration_keys) > 0:
        for integration_key in integration_keys:
            if integration_key is None: continue
            try:
                data={
                    'routing_key': integration_key,
                    'event_action': 'resolve',
                    'dedup_key' : dedup_key,
                    'payload': {
                        "summary" : f"{source} : Incident {dedup_key} resolved.",
                        "source" : source,
                        "severity" : "info"
                    }
                }
                pypd.EventV2.create(data=data)
            except Exception as e:
                bt.logging.error(f"FAILED TO RESOLVE PAGERDUTY ALERT : {str(e)}\nDATA: {data}")           