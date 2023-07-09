# -*- coding: utf-8 -*-
import torch
def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time.bool()
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = torch.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time

def _estimate_concordance_index(event_indicator, event_time, estimate, tied_tol=1e-8):
    order = torch.argsort(event_time) #event_time

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise Exception(
            "Data has no comparable pairs, cannot estimate concordance index.")

    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        con = torch.div(torch.log(1/(1+torch.exp(( est_i - est )/1)) ),torch.log(torch.tensor(2.0))) + 1 
        # con = 1/(1+torch.exp(( est_i - est )/5)) 
        numerator += torch.sum(con)
        denominator += torch.sum(mask.double())

    cindex = numerator / denominator
    return cindex

def _estimate_concordance_index2(event_indicator, event_time, estimate, tied_tol=1e-8):
    order = torch.argsort(event_time) #event_time

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise Exception(
            "Data has no comparable pairs, cannot estimate concordance index.")

    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        con = 1/(1+torch.exp(( est_i - est )/1)) 
        numerator += torch.sum(con)
        denominator += torch.sum(mask.double())

    cindex = numerator / denominator
    return cindex


def _estimate_concordance_index_true(event_indicator, event_time, estimate, tied_tol=1e-8):
    order = torch.argsort(event_time) #event_time

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise Exception(
            "Data has no comparable pairs, cannot estimate concordance index.")

    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        con = 1/(1+torch.exp(( est_i - est )/0.01)) 
        numerator += torch.sum(con)
        denominator += torch.sum(mask.double())

    cindex = numerator / denominator
    return cindex


