import extract_signature


if __name__ == '__main__':
    lines = ''' Sehr geehrte Damen und Herren,

ich habe am Dienstag, den 20.11.18 gegen 21:00 Uhr eine bestellunge abgeschickt und habe leider noch keine Bestellbestätigung erhalten. Ist die Bestellung bei Ihnen eingegangen oder gab es Probleme bei der übermittlung ?

Ich bitte um schnelle Antwort, damit ich ggf. heute die 35% Aktion nutzen kann.

Danke im Voraus

mit freundlichen Grüßen 
Marius Gathemann'''  ####Grüßen

    body, signature = extract_signature.extract_signature(lines)
    print('body: {}\n\nsignature: {}'.format(body, signature))