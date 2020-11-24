import json
from config import graph




f = open(r'C:\Users\lenovo\Desktop\medium.txt','r',encoding='utf-8')
text = json.load(f)



for article in text:
    print(article['id'])
    graph.run('MERGE(p: Article{Id:"%s",name: "%s",name_zh:"%s",level:"%s",definition:"%s",definition_zh:"%s"})' % (article['id'],article['name'],article['name_zh'],article['level'],article['definition'],article['definition_zh']))

    if article['child_nodes']:
        for child in article['child_nodes']:
            graph.run(
                'MATCH(a: Article), (b: Article) \
                WHERE a.Id="%s" AND b.Id="%s"\
                CREATE(a)-[r:%s{child: "%s"}]->(b)\
                RETURN r' % (str(article['id']), str(child), 'child', 'child')
            )
    if article['experts']:
        for expert in article['experts']:
            try:
                graph.run('MERGE(p: Expert{Id:"%s",name: "%s",name_zh: "%s",position:"%s",h_index:"%s"})' % (expert['id'], expert['name'], expert['name_zh'], expert['position'], str(expert['h_index'])))
                graph.run(
                    'MATCH(a: Article), (cc: Expert) \
                    WHERE a.Id="%s" AND cc.Id="%s"\
                    CREATE(a)-[r:%s{expert: "%s"}]->(cc)\
                    RETURN r' % (article['id'], expert['id'] ,'expert', 'expert')
                )
            except Exception as e:
                print(e)
            if expert['interests']:
                for interest in expert['interests']:
                    try:
                        graph.run('MERGE(p: Interest{interest:"%s"})' % (interest))

                        graph.run(
                            'MATCH(a: Expert), (cc: Interest) \
                            WHERE a.Id="%s" AND cc.interest="%s"\
                            CREATE(a)-[r:%s{interest: "%s"}]->(cc)\
                            RETURN r' % (expert['id'], interest, 'interest', 'interest')
                        )
                    except Exception as e:
                        print(e)


    '''
    for publication in article['publications']:
        if '"' in publication['title']:
            publication['title'] = publication['title'].replace('\"','\'')
        graph.run('MERGE(p: Publication{Id:"%s",title: "%s"})' % (publication['id'],publication['title']))
        graph.run(
            "MATCH(a: Article), (cc: Publication) \
            WHERE a.Id='%s' AND cc.Id='%s'\
            CREATE(a)-[r:%s{publication: '%s'}]->(cc)\
            RETURN r" % (article['id'], publication['id'], 'publication', 'publication')
        )
    '''

