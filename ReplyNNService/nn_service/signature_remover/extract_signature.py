from nn_service.signature_remover import constants

from nn_service import helper as helper

def _mark_candidate_indexes(lines, candidate, use_nn=False):
    """Mark candidate indexes with markers

    Markers:

    * c - line that could be a signature line
    * l - long line
    * d - line that starts with dashes but has other chars as well

    _mark_candidate_lines(['Some text', '', '-', 'Bob'], [0, 2, 3])
    'cdc'
    """
    # at first consider everything to be potential signature lines
    markers = list('c' * len(candidate))

    if not use_nn:
        # mark lines starting from bottom up
        for i, line_idx in reversed(list(enumerate(candidate))):
            if len(lines[line_idx].strip()) > constants.TOO_LONG_SIGNATURE_LINE:
                markers[i] = 'l'
            else:
                line = lines[line_idx].strip()
                if line.startswith('-') and line.strip("-"):
                    markers[i] = 'd'
    else:
        # mark lines starting from bottom up
        for i, line_idx in reversed(list(enumerate(candidate))):
            line = lines[line_idx].strip()
            classes, probs = helper.classify(line, 'sv')
            # print('classes, probs for {}'.format(lines[i]), classes, probs)
            if classes[0] == 'SIGNATURE':
                markers[i] = 'c'
            else:
                markers[i] = 'l'

            if line.startswith('-') and line.strip("-"):
                markers[i] = 'd'
            # print(line, '\t\t\t', markers[i])

    # print('signature markers', "".join(markers))
    return "".join(markers)


def _process_marked_candidate_indexes(candidate, markers):
    """
    Run regexes against candidate's marked indexes to strip
    signature candidate.

    >>> _process_marked_candidate_indexes([9, 12, 14, 15, 17], 'clddc')
    [15, 17]
    """
    match = constants.RE_SIGNATURE_CANDIDATE.match(markers[::-1])
    return candidate[-match.end('candidate'):] if match else []


def get_signature_candidate(lines, use_nn=False):
    """Return lines that could hold signature

    The lines should:

    * be among last SIGNATURE_MAX_LINES non-empty lines.
    * not include first line
    * be shorter than TOO_LONG_SIGNATURE_LINE
    * not include more than one line that starts with dashes
    """
    # non empty lines indexes
    non_empty = [i for i, line in enumerate(lines) if line.strip()]

    # if message is empty or just one line then there is no signature
    if len(non_empty) <= 1:
        return []

    # we don't expect signature to start at the 1st line
    candidate = non_empty[1:]
    # signature shouldn't be longer then SIGNATURE_MAX_LINES
    candidate = candidate[-constants.SIGNATURE_MAX_LINES:]

    markers = _mark_candidate_indexes(lines, candidate, use_nn=use_nn)
    candidate = _process_marked_candidate_indexes(candidate, markers)

    # get actual lines for the candidate instead of indexes
    if candidate:
        candidate = lines[candidate[0]:]
        return candidate

    return []


def get_delimiter(msg_body):
    delimiter = constants.RE_DELIMITER.search(msg_body)
    if delimiter:
        delimiter = delimiter.group()
    else:
        delimiter = '\n'

    return delimiter


def extract_signature(msg_body, use_nn=False):
    """
    Analyzes message for a presence of signature block (by common patterns)
    and returns tuple with two elements: message text without signature block
    and the signature itself.
    extract_signature('Hey man! How r u? \n\n--\nRegards,\nRoman')
    ('Hey man! How r u?', '--\nRegards,\nRoman')
    extract_signature('Hey man!')
    ('Hey man!', None)
    """
    try:
        # identify line delimiter first
        delimiter = get_delimiter(msg_body)

        # make an assumption
        stripped_body = msg_body.strip()

        # strip off phone signature
        phone_signature = constants.RE_PHONE_SIGNATURE.search(msg_body)
        if phone_signature:
            stripped_body = stripped_body[:phone_signature.start()]
            phone_signature = phone_signature.group()

        # decide on signature candidate
        lines = stripped_body.splitlines()
        candidate = get_signature_candidate(lines, use_nn=use_nn)
        candidate = delimiter.join(candidate)

        # try to extract signature
        signature = constants.RE_SIGNATURE.search(candidate)
        if not signature:
            return stripped_body.strip(), phone_signature
        else:
            signature = signature.group()
            # when we splitlines() and then join them
            # we can lose a new line at the end
            # we did it when identifying a candidate
            # so we had to do it for stripped_body now
            stripped_body = delimiter.join(lines)
            stripped_body = stripped_body[:-len(signature)]

            if phone_signature:
                signature = delimiter.join([signature, phone_signature])

            return (stripped_body.strip(),
                    signature.strip())
    except Exception as e:
        print('ERROR extracting signature', e)
        return msg_body, None
