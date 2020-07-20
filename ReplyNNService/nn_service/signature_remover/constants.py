import regex as re

RE_SIGNATURE_CANDIDATE = re.compile(r'''
    (?P<candidate>c+d)[^d]
    |
    (?P<candidate>c+d)$
    |
    (?P<candidate>c+)
    |
    (?P<candidate>d)[^d]
    |
    (?P<candidate>d)$
''', re.I | re.X | re.M | re.S)

RE_SIGNATURE = re.compile(r'''
               (
                   (?:
                       ^[\s]*--*[\s]*[a-z \.]*$
                       |
                       ^[\s]*[/ ]{1,2}[\s]*[a-z \.]*$
                       |
                       ^([\w]*[,.]?[ ])?thanks[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?thank[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?regards[ \w]*[\s,!.]*
                       |
                       ^cheers[ \w]*[\s,!.]*
                       |
                       ^best[ a-z]*[\s,!]*$
                       |
                       ^[\s]?met[ \w]*[\s,!.]*
                       |
                       ^[\s]?vänliga[ \w]*[\s,!.]*
                       |
                       ^groet[ \w]*[\s,!.]*
                       |
                       ^varma[ \w]*[\s,!.]*
                       |
                       ^ha[ \w]*[\s,!.]*
                       |
                       ^vielen[ \w]*[\s,!.]*
                       |
                       ^[\s]?mvh[ \w]*[\s,!.]*
                       |
                       ^.*?[\s]?mvh[ \w]{1,3}[\s,!.]*
                       |
                       ^[\s]?mit[ \w]*[\s,!.]*
                       |
                       ^[\s]?cordialement[ \w]*[\s,!.]*
                       |
                       ^[\s]?grüße[ \w]*[\s,!.]*
                       |
                       ^[\s]?med[ \w]*[\s,!.]*
                       |
                       ^[\s]?[med vänlig hälsning][ \w]{1,3}[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?merci[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?cordiali[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?ystävällisin[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?gracias[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?takk[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?vriendelijke[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?grüße[ \w]*[\s,!.]*
                       |
                       ^([\w]*[,.]?[ ])?cordiali[ \w]*[\s,!.]*
                   )
                   .*
               )
               ''', re.I | re.X | re.M | re.S)


"""|
                       ^[\s]?tack[ \w]*[\s,!.]*
                       |
                       ^[\s]?bästa[ \w]*[\s,!.]*
                       """


# signatures appended by phone email clients
RE_PHONE_SIGNATURE = re.compile(r'''
               (
                   (?:
                       ^[\s]?sent[ ]{1}from[ ]{1}my[\s,!\w]*$
                       |
                       ^sent[ ]from[ ]Mailbox[ ]for[ ]iPhone.*$
                       |
                       ^sent[ ]([\S]*[ ])?from[ ]my[ ]BlackBerry.*$
                       |
                       ^Enviado[ ]desde[ ]mi[ ]([\S]+[ ]){0,2}BlackBerry.*$
                       |
                       ^[\s]?Sendt[ ]fra[ ][\s,!\w]*$
                       |
                       ^[\s]?Skickat[ ]från[ ]min[\s,!\w]*$
                   )
                   .*
               )
               ''', re.I | re.X | re.M | re.S)

RE_DELIMITER = re.compile('\r?\n')

SIGNATURE_MAX_LINES = 20 # v11
TOO_LONG_SIGNATURE_LINE = 60
delimiter = '\n'
