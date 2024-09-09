package java_generative_ai;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SentimentAnalysis {

    public static void main(String[] args) throws Exception {
        // Carregar o conjunto de dados (arquivo .arff com textos rotulados)
        System.out.println("Carregando o conjunto de dados...");
        DataSource source = new DataSource("C:\\Users\\HOME\\Downloads\\Java_GenerativeAI\\Java_Generative_AI\\src\\java_generative_ai\\sentiment_data.arff");
        Instances dataset = source.getDataSet();
        System.out.println("Conjunto de dados carregado com sucesso!");

        // Definir qual atributo será classificado (último no conjunto de dados)
        System.out.println("Definindo o atributo que será classificado...");
        dataset.setClassIndex(dataset.numAttributes() - 1);
        System.out.println("Atributo de classe definido: " + dataset.classAttribute().name());

        // Exibir o texto original antes da conversão
        System.out.println("Texto original: " + dataset.get(0));

        // Transformar atributos de texto em vetores de palavras
        System.out.println("Convertendo atributos de texto em vetores de palavras...");
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataset); // Configurar o filtro para o formato dos dados

        // Aplicar o filtro ao conjunto de dados
        Instances filteredData = Filter.useFilter(dataset, filter);
        System.out.println("Atributos de texto convertidos com sucesso!");

        // Criar e treinar o classificador NaiveBayes
        System.out.println("Treinando o modelo NaiveBayes...");
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(filteredData);
        System.out.println("Modelo treinado com sucesso!");

        // Exemplo de nova instância para classificar
        System.out.println("Classificando um novo exemplo de texto...");
        Instance newText = filteredData.get(0); // Simulando um novo texto para classificação
        double label = classifier.classifyInstance(newText);

        // Exibir resultado da classificação
        System.out.println("Classificação prevista: " + filteredData.classAttribute().value((int) label));
    }
}
